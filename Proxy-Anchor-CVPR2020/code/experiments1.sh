# python train2.py --gpu-id 1 \
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
#                 --remark img224


# python train2.py --gpu-id 1 \
#                 --loss Arcface \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --backbone-model resnet101 \
#                 --remark resnet101

# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \``
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --backbone-model resnet101 \
#                 --remark resnet101

# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --backbone-model resnet101 \
#                 --remark resnet101

# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --IPC 3 \
#                 --backbone-model resnet101 \
#                 --remark IPC3-resnet101

# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 5 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --backbone-model resnet101 \
#                 --remark warm5-resnet101

# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 5 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --IPC 3 \
#                 --backbone-model resnet101 \
#                 --remark warm5-IPC3-resnet101


# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --resize 292 \
#                 --crop 256 \
#                 --backbone-model resnet101 \

# python train2.py --gpu-id 1 \
#                 --loss Proxy_Anchor \
#                 --model dolg \
#                 --embedding-size 512 \
#                 --batch-size 120 \
#                 --dataset cub \
#                 --warm 0 \
#                 --lr-decay-step 5 \
#                 --epochs 30 \
#                 --lr 1e-3 \
#                 --optimizer sgd \
#                 --bn-freeze 0 \
#                 --resize 292 \
#                 --crop 256 \
#                 --IPC 3 \
#                 --backbone-model resnet101 \
#                 --remark img256-IPC3

# python train3.py --gpu-id 1 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --loss-lambda 5 \
#                  --remark G


# python train3.py --gpu-id 1 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config G \
#                  --loss-lambda 10 \
#                  --remark G-lambda10

# python train3.py --gpu-id 1 \
#                  --epochs 35 \
#                  --model cgdolg4 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --loss-lambda 1 \
#                  --remark MG

# python train3.py --gpu-id 1 \
#                  --epochs 35 \
#                  --model cgdolg3 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 7 \
#                  --warm 3 \
#                  --backbone-model resnet50 \
#                  --gd-config GMS \
#                  --loss-lambda 1 \
#                  --remark GMS-lambda1-warm3

# python train3.py --gpu-id 1 \
#                  --epochs 35 \
#                  --model cgdolg5 \
#                  --loss Proxy_Anchor \
#                  --optimizer sgd \
#                  --lr 1e-3 \
#                  --lr-decay-step 5 \
#                  --warm 0 \
#                  --backbone-model resnet50 \
#                  --gd-config MG \
#                  --loss-lambda 1 \
#                  --remark MG-lambda1

# python train3.py --gpu-id 1 \
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
#                  --remark MG-lambda1-l2norm-nonaux

# python train3.py --gpu-id 1 \
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
#                  --remark MG-nonaux-l2

# python train3.py --gpu-id 1 \
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
#                  --remark G-nonaux-l2

# python train3.py --gpu-id 1 \
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
#                  --remark MSG-nonaux-l2

python train3.py --gpu-id 1 \
                 --epochs 35 \
                 --model cgdolg5 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model seresnet50 \
                 --gd-config MG \
                 --auxiliary-loss 0 \
                 --resize 256 \
                 --crop 224 \
                 --remark SERes50-MG-224size

python train3.py --gpu-id 1 \
                 --epochs 35 \
                 --model cgdolg5 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model seresnet50 \
                 --gd-config MG \
                 --auxiliary-loss 0 \
                 --resize 292 \
                 --crop 256 \
                 --remark SERes50-MG-256size

python train3.py --gpu-id 1 \
                 --epochs 35 \
                 --model cgdolg5 \
                 --loss Proxy_Anchor \
                 --optimizer sgd \
                 --lr 1e-3 \
                 --lr-decay-step 5 \
                 --warm 0 \
                 --backbone-model seresnet50 \
                 --gd-config MG \
                 --auxiliary-loss 0 \
                 --resize 361 \
                 --crop 324 \
                 --remark SERes50-MG-324size