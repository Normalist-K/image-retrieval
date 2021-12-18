# python train.py --gpu-id 0 \
#                 --loss Proxy_Anchor \
#                 --model resnet50 \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 5 \
#                 --bn-freeze 1 \
#                 --lr-decay-step 5 \
#                 --epochs 25 \
#                 --lr 1e-3 \
#                 --optimizer sgd

# python train2.py --gpu-id 0 \
#                 --loss Arcface \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 25 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0

# python train2.py --gpu-id 0 \
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
#                 --bn-freeze 0

# python train2.py --gpu-id 0 \
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
#                 --resize 292 \
#                 --crop 256 \

# python train2.py --gpu-id 0 \
#                 --loss Arcface \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 25 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0
#                 --IPC 3 \
#                 --remark IPC3
#=== Arcface는 sampling에 영향을 거의 받지 않는다.


# python train2.py --gpu-id 0 \
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
#                 --resize 292 \
#                 --crop 256 \
#                 --IPC 3 \
#                 --remark img256-IPC3


# python train2.py --gpu-id 0 \
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
#                 --backbone-model resnet50_simclr \
#                 --remark r50-simclr


# python train2.py --gpu-id 0 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 35 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --backbone-model resnet50_simclr \
#                 --remark r50-simclr-e35

# python train2.py --gpu-id 2 \
#                 --loss Proxy_Anchor \
#                 --model cgdolg2 \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 35 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --gd-config SG \
#                 --remark SG

# python train3.py --gpu-id 0 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --loss-lambda 10 \
#                  --remark MG-lambda10

# python train3.py --gpu-id 0 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 7 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --loss-lambda 1 \
#                  --remark MG-lambda1-warm3

# python train3.py --gpu-id 0 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 7 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --loss-lambda 1 \
#                  --remark G-lambda1-warm3

# python train3.py --gpu-id 0 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 7 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config GS \
#                  --loss-lambda 1 \
#                  --remark GS-lambda1-warm3

# python train3.py --gpu-id 0 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 7 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config GM \
#                  --loss-lambda 1 \
#                  --remark GM-lambda1-warm3

# python train.py --gpu-id 0 \
#                 --loss Proxy_Anchor \
#                 --model resnet50 \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 5 \
#                 --bn-freeze 1 \
#                 --lr-decay-step 5 \
#                 --epochs 25 \
#                 --lr 1e-3 \
#                 --optimizer sgd

# python train2.py --gpu-id 0 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 35 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0



# """
# # implement
# CGD + DolG

# # backbone model
# Inception-BN
# ResNet-101
# efficientnet

# # config

# """