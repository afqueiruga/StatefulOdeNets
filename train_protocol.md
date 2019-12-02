
# MNIST: LeNetLike

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train.py --name mnist --epochs 90 --arch LeNetLike --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 80 --weight-decay 5e-4

# CIFAR10: AlexNetLike 

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train.py --name cifar10 --epochs 90 --arch AlexLike --lr 0.02 --lr-decay 0.2 --lr-schedule normal --lr-decay-epoch 30 60 80

# CIFAR10: ResNet

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train.py --name cifar10 --epochs 120 --arch ResNet --lr 0.1 --lr-decay 0.1 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4

# CIFAR10: WideResNet

## Baseline Training
> export CUDA_VISIBLE_DEVICES=0; python train.py --name cifar10 --epochs 120 --arch WideResNet --lr 0.1 --lr-decay 0.1 --lr-schedule normal --lr-decay-epoch 30 60 90 --batch-size 128 --test-batch-size 200 --weight-decay 5e-4 --depth 34 --widen_factor 4





