# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 25 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --IPC 3 \
#                 --remark IPC3

# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model cgdolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 25 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \

# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 5 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-4 \
#                 --optimizer adamw \
#                 --bn-freeze 0 \
#                 --backbone-model resnet50 \
#                 --remark resnet50-warm5

# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model cgdolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 35 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --gd-config MSG \
#                 --remark MSG

# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model cgdolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 35 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --gd-config MSG \
#                 --remark MG

# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model cgdolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 35 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --gd-config MSG \
#                 --remark MS

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --loss-lambda 5 \
#                  --remark MG

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --loss-lambda 1 \
#                  --remark MG-lambda1

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --loss-lambda 1 \
#                  --remark G-lambda1

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config GM \
#                  --loss-lambda 1 \
#                  --remark GM-lambda1

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config GS \
#                  --loss-lambda 1 \
#                  --remark GS-lambda1

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg4 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --auxiliary-loss 0 \
#                  --remark MG-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --loss-lambda 1 \
#                  --auxiliary-loss 0 \
#                  --remark G-lambda1-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --loss-lambda 1 \
#                  --remark G-lambda1


# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MSG \
#                  --auxiliary-loss 0 \
#                  --remark MSG-lambda1-l2norm-nonaux


# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --auxiliary-loss 0 \
#                  --remark G-lambda1-l2norm-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config GS \
#                  --auxiliary-loss 0 \
#                  --remark GS-lambda1-l2norm-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --auxiliary-loss 0 \
#                  --remark MG-lambda1-l2norm-nonaux-warm3

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config MSG \
#                  --auxiliary-loss 0 \
#                  --remark MSG-lambda1-l2norm-nonaux-warm3

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config S \
#                  --auxiliary-loss 0 \
#                  --remark S-lambda1-l2norm-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config M \
#                  --auxiliary-loss 0 \
#                  --remark M-lambda1-l2norm-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MS \
#                  --auxiliary-loss 0 \
#                  --remark MS-lambda1-l2norm-nonaux

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --auxiliary-loss 2 \
#                  --loss-lambda 1 \
#                  --remark MG-l2norm-aux2

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --auxiliary-loss 2 \
#                  --loss-lambda 1 \
#                  --remark G-l2norm-aux2
                 
# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MSG \
#                  --auxiliary-loss 2 \
#                  --loss-lambda 1 \
#                  --remark MSG-l2norm-aux2

#esolution Resize List : 256, 292, 361, 512
# Resolution Crop List: 224, 256, 324, 448

python train3.py --gpu-id 2 \
                 --epochs 35 \
                 --model cgdolg5 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model resnet50 \
                 --gd-config MG \
                 --auxiliary-loss 0 \
                 --resize 292 \
                 --crop 256 \
                 --remark MG-256size

# python train3.py --gpu-id 2 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --auxiliary-loss 0 \
#                  --resize 361 \
#                  --crop 324 \
#                  --remark MG-324size