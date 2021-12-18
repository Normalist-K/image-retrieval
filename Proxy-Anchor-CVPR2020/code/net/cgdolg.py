import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset


class GlobalDescriptor(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, x):
        assert x.dim() == 4, 'the input tensor of GlobalDescriptor must be the shape of [B, C, H, W]'
        if self.p == 1:
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            x = F.avg_pool2d(x.clamp(min=1e-6).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
            return x.squeeze(-1).squeeze(-1)

    def extra_repr(self):
        return f"p={self.p}"


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, 'the input tensor of L2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)


class CGD(nn.Module):
    def __init__(self, gd_config='MG', feature_dim=1024, num_classes=200):
        super(CGD,self).__init__()

        self.n = len(gd_config)
        k = feature_dim // self.n

        global_descriptors = []
        main_modules = []

        dim_count = feature_dim
        for gd in gd_config:
            if gd == 'S':
                p = 1
            elif gd == 'M':
                p = float('inf')
            else:
                p = 3
            dim_count -= k
            if (dim_count > 0) and (dim_count < k):
                k += dim_count
            global_descriptors.append(GlobalDescriptor(p=p))
            main_modules.append(nn.Sequential(nn.Linear(feature_dim, k, bias=False), L2Norm()))
        self.global_descriptors = nn.ModuleList(global_descriptors)
        self.main_moduels = nn.ModuleList(main_modules)

        self.auxiliary_module = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, num_classes, bias=True))

    def forward(self, x):
        gds = []
        for i in range(self.n):
            gd = self.global_descriptors[i](x)
            if i == 0:
                classes = self.auxiliary_module(gd)
            gd = self.main_moduels[i](gd)
            gds.append(gd)
        global_descriptor = F.normalize(torch.cat(gds, dim=-1), dim=-1)
        return global_descriptor, classes


class CGD2(nn.Module):
    def __init__(self, gd_config='SG', feature_dim=1024):
        super(CGD2,self).__init__()

        n = len(gd_config)

        global_descriptors = []
        for gd in gd_config:
            if gd == 'S':
                p = 1
            elif gd == 'M':
                p = float('inf')
            else:
                p = 3
            global_descriptors.append(
                nn.Sequential(
                    GlobalDescriptor(p=p), 
                    L2Norm()
                    )
                )
        self.global_descriptors = nn.ModuleList(global_descriptors)
        self.linear = nn.Linear(feature_dim*n, feature_dim, bias=False)

    def forward(self, x):
        gds = []
        for global_descriptor in self.global_descriptors:
            gd = global_descriptor(x)
            gds.append(gd)
        global_descriptor = torch.cat(gds, dim=-1)
        global_descriptor = F.normalize(self.linear(global_descriptor), dim=-1)
        return global_descriptor


class MultiAtrous(nn.Module):
    def __init__(self, in_channel, out_channel, size, dilation_rates=[3, 6, 9]):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel/4),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(out_channel/4), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode='bilinear')
        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat


class DolgLocalBranch(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=2048, image_size=512):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, hidden_channel, size=int(image_size/8))
        self.conv1x1_1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        local_feat = self.multi_atrous(x)

        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        projection = projection / \
            (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)

class CGDolgNet4(nn.Module):
    def __init__(self,
                 model_name='resnet50', 
                 pretrained=True, 
                 input_dim=3, 
                 hidden_dim=1024, 
                 output_dim=512, 
                 image_size=224,
                 gd_config='MG'):
        super().__init__()

        if model_name == 'resnet101':
            model_name = 'gluon_resnet101_v1b'

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 4)
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_dim, 2048, image_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cgd = CGD(gd_config)
        self.fc_1 = nn.Linear(2048, hidden_dim)
        self.fc_2 = nn.Linear(int(2*hidden_dim), output_dim)

    def forward(self, x):
        output = self.model(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat, classes = self.cgd(output[1])  # ,1024
        global_feat = self.fc_1(global_feat)
        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat, classes

class CGDolgNet3(nn.Module):
    def __init__(self,
                 model_name='resnet50', 
                 pretrained=True, 
                 input_dim=3, 
                 hidden_dim=1024, 
                 output_dim=512, 
                 image_size=224,
                 gd_config='MG'):
        super().__init__()

        if model_name == 'resnet101':
            model_name = 'gluon_resnet101_v1b'

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 3)
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_dim, 2048, image_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cgd = CGD(gd_config)
        self.fc_1 = nn.Linear(1024, hidden_dim)
        self.fc_2 = nn.Linear(int(2*hidden_dim), output_dim)

    def forward(self, x):
        output = self.model(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat, classes = self.cgd(output[1])  # ,1024

        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat, classes




class CGDolgNet(nn.Module):
    def __init__(self,
                 model_name='resnet50', 
                 pretrained=True, 
                 input_dim=3, 
                 hidden_dim=1024, 
                 output_dim=512, 
                 image_size=224,
                 gd_config='SG'):
        super().__init__()

        if model_name == 'resnet101':
            model_name = 'gluon_resnet101_v1b'

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 3)
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_dim, 2048, image_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cgd = CGD(gd_config)
        self.fc_1 = nn.Linear(1024, hidden_dim)
        self.fc_2 = nn.Linear(int(2*hidden_dim), output_dim)

    def forward(self, x):
        output = self.model(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.cgd(output[1]).squeeze())  # ,1024

        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat

class CGDolgNet2(nn.Module):
    def __init__(self,
                 model_name='resnet50', 
                 pretrained=True, 
                 input_dim=3, 
                 hidden_dim=1024, 
                 output_dim=512, 
                 image_size=224,
                 gd_config='SG'):
        super().__init__()

        if model_name == 'resnet101':
            model_name = 'gluon_resnet101_v1b'

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 3)
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_dim, 2048, image_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cgd = CGD2(gd_config)
        self.fc_1 = nn.Linear(1024, hidden_dim)
        self.fc_2 = nn.Linear(int(2*hidden_dim), output_dim)

    def forward(self, x):
        output = self.model(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.cgd(output[1]).squeeze())  # ,1024

        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat
