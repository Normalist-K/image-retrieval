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

python train3.py --gpu-id 2 \
                 --epochs 35 \
                 --model cgdolg3 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model resnet50 \
                 --gd-config MG \
                 --loss-lambda 1 \
                 --remark MG-lambda1

python train3.py --gpu-id 2 \
                 --epochs 35 \
                 --model cgdolg3 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model resnet50 \
                 --gd-config G \
                 --loss-lambda 1 \
                 --remark G-lambda1

python train3.py --gpu-id 2 \
                 --epochs 35 \
                 --model cgdolg3 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model resnet50 \
                 --gd-config GM \
                 --loss-lambda 1 \
                 --remark GM-lambda1

python train3.py --gpu-id 2 \
                 --epochs 35 \
                 --model cgdolg3 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model resnet50 \
                 --gd-config GS \
                 --loss-lambda 1 \
                 --remark GS-lambda1