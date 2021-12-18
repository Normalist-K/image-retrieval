import datetime as dt
import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.dolg import *
from net.cgdolg import *
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'  
    + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
    default='../logs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 120, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 60, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'bn_inception',
    help = 'Model for training'
)
parser.add_argument('--loss', default = 'Proxy_Anchor',
    help = 'Criterion for training'
)
parser.add_argument('--optimizer', default = 'adamw',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 1e-4, type =float,
    help = 'Learning rate setting'
)
parser.add_argument('--weight-decay', default = 1e-4, type =float,
    help = 'Weight decay setting'
)
parser.add_argument('--lr-decay-step', default = 10, type =int,
    help = 'Learning decay step setting'
)
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
    help = 'Learning decay gamma setting'
)
parser.add_argument('--alpha', default = 32, type = float,
    help = 'Scaling Parameter setting'
)
parser.add_argument('--mrg', default = 0.1, type = float,
    help = 'Margin parameter setting'
)
parser.add_argument('--IPC', type = int,
    help = 'Balanced sampling, images per class'
)
parser.add_argument('--warm', default = 1, type = int,
    help = 'Warmup training epochs'
)
parser.add_argument('--bn-freeze', default = 0, type = int,
    help = 'Batch normalization parameter freeze'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--resize', default = 256, type=int,
    help = 'Transform resize img size'
)
parser.add_argument('--crop', default = 224, type=int,
    help = 'Transform crop img size'
)
parser.add_argument('--backbone-model', default = 'resnet50',
    help = 'Backbone model for DolG'
)
parser.add_argument('--gd-config', default='SG',
    help = 'CGD gd configuration : combination of "M", "S", "G"'
)
parser.add_argument('--smoothing', default=0.1, type=float,
    help = 'smoothing value for label smothing'
)
parser.add_argument('--temperature', default=0.5, type=float,
    help = 'temperature scaling used in softmax cross-entropy loss'
)
parser.add_argument('--loss-lambda', default=5, type=float,
    help = 'temperature scaling used in softmax cross-entropy loss'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

date_time = dt.datetime.now()
# Directory for Log
LOG_DIR = args.LOG_DIR + f'/logs_cgdolg3/{date_time.day}.{date_time.hour}h{date_time.minute}m-{args.remark}'
# Wandb Initialization
wandb.init(project="cv-project", notes=LOG_DIR, name=f'{args.model}-{args.remark}')
wandb.config.update(args)

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
trn_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'train',
        transform = dataset.utils.make_transform(
            is_train = True, 
            is_inception = (args.model == 'bn_inception'),
            resize= args.resize,
            crop = args.crop
        ))

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )
    print('Balanced Sampling')
    
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )
    print('Random Sampling')

ev_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'eval',
        transform = dataset.utils.make_transform(
            is_train = False, 
            is_inception = (args.model == 'bn_inception'),
            resize= args.resize,
            crop = args.crop
        ))

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)
    
nb_classes = trn_dataset.nb_classes()

# Backbone Model
if args.model == 'cgdolg3':
    model = CGDolgNet3(model_name=args.backbone_model, image_size=args.crop, gd_config=args.gd_config)
elif args.model == 'cgdolg4':
    model = CGDolgNet4(model_name=args.backbone_model, image_size=args.crop, gd_config=args.gd_config)
# if args.model == 'dolg':
#     model = DolgNet(model_name=args.backbone_model, image_size=args.crop)
# elif args.model == 'cgdolg':
#     model = CGDolgNet(model_name=args.backbone_model, image_size=args.crop, gd_config=args.gd_config)
# elif args.model == 'cgdolg2':
#     model = CGDolgNet2(model_name=args.backbone_model, image_size=args.crop, gd_config=args.gd_config)
# if args.model.find('googlenet')+1:
#     model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
# elif args.model.find('bn_inception')+1:
#     model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
# elif args.model.find('resnet18')+1:
#     model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
# elif args.model.find('resnet50')+1:
#     model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
# elif args.model.find('resnet101')+1:
#     model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

# DML Losses
if args.loss == 'Proxy_Anchor':
    metric_criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args.sz_embedding, mrg = args.mrg, alpha = args.alpha).cuda()
elif args.loss == 'Proxy_NCA':
    metric_criterion = losses.Proxy_NCA(nb_classes = nb_classes, sz_embed = args.sz_embedding).cuda()
elif args.loss == 'MS':
    metric_criterion = losses.MultiSimilarityLoss().cuda()
elif args.loss == 'Contrastive':
    metric_criterion = losses.ContrastiveLoss().cuda()
elif args.loss == 'Triplet':
    metric_criterion = losses.TripletLoss().cuda()
elif args.loss == 'NPair':
    metric_criterion = losses.NPairLoss().cuda()
elif args.loss == 'Arcface':
    metric_criterion = losses.ArcFace(
            in_features=512,
            out_features=200,
            scale_factor=30,
            margin=0.15,
            criterion=nn.CrossEntropyLoss()
        ).cuda()

class_criterion = losses.LabelSmoothingCrossEntropyLoss(args.smoothing, args.temperature)

# Train Parameters
if args.model in ['dolg', 'cgdolg', 'cgdolg2', 'cgdolg3', 'cgdolg4']:
    param_groups = [{'params': model.parameters() if args.gpu_id != -1 else model.module.parameters(), 'lr':float(args.lr) * 1},]
else:
    param_groups = [
        {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else 
                    list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args.lr) * 1},
    ]
if args.loss == 'Proxy_Anchor':
    param_groups.append({'params': metric_criterion.proxies, 'lr':float(args.lr) * 100})

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list, class_losses_list, metric_losses_list = [], [], []
best_recall=[0]
best_epoch = 0

for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch, class_losses, metric_losses = [], [], []
    
    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        if args.model in ['dolg', 'cgdolg', 'cgdolg2', 'cgdolg3']:
            if epoch == 0:
                for param in list(model.model.parameters()):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(model.model.parameters()):
                    param.requires_grad = True
        else:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
            else:
                unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:                         
        descriptor, classes = model(x.squeeze().cuda())
        metric_loss = metric_criterion(descriptor, y.squeeze().cuda())
        class_loss = class_criterion(classes, y.squeeze().cuda())
        loss = (metric_loss + (class_loss * args.loss_lambda)) / 2
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.loss == 'Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(metric_criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        class_losses.append(class_loss.data.cpu().numpy())
        metric_losses.append(metric_loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
        
    losses_list.append(np.mean(losses_per_epoch))
    class_losses_list.append(np.mean(class_losses))
    metric_losses_list.append(np.mean(metric_losses))
    wandb.log({'loss': losses_list[-1],
               'c_loss': class_losses_list[-1],
               'm_loss': metric_losses_list[-1]},                
              step=epoch)
    scheduler.step()
    
    if(epoch >= 0):
        with torch.no_grad():
            print("**Evaluating...**")
            Recalls = utils.evaluate_cos(model, dl_ev)
                
        # Logging Evaluation Score
        for i in range(6):
            wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)
        
        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists('{}'.format(LOG_DIR)):
                os.makedirs('{}'.format(LOG_DIR))
            torch.save({'model_state_dict':model.state_dict()}, '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
            with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                for i in range(6):
                    f.write("Best Recall@{}: {:.4f}\n".format(2**i, best_recall[i] * 100))

    
