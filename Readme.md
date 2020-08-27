# Continuous-in-Depth Neural Networks

Introduction (todo).

This directory contains the implementation of ContinuousNet to accompany the preprint
> [Queiruga, Alejandro F., N. Benjamin Erichson, Dane Taylor and Michael W. Mahoney. “Continuous-in-Depth Neural Networks.” ArXiv abs/2008.02389 (2020)](https://arxiv.org/abs/2008.02389)

<img src="https://github.com/erichson/data/blob/master/img/ContinuousNet_overview.png" width="750">


## Get Started

Just clone the ContinuousNet repository to your local system and you are ready to go:
```
git clone https://github.com/afqueiruga/ContinuousNet
```

Also, you need a fork of `torchdiffeq` that is slightly modified in order to expose additional solver options: 
```
pip install git+https://github.com/afqueiruga/torchdiffeq
```

Note, this implementation best works with [PyTorch 1.6](https://pytorch.org/).

## Training

ContinuousNets can be trained similar to ResNets via a command line interface:
```
python3 cli.py [--dataset] [--batch-size] [--test-batch-size] [--epochs] [--lr] [--lr-decay] [--lr-decay-epoch] [--weight-decay] [--batch-norm] [--device] [--seed]

standard arguments:
--dataset                   you can train on CIFAR10, or CIFAR100 (default: CIFAR10)	
--batch-size                training batch size (default: 128)
--test-batch-size           testing batch size (default:256)
--epochs                    total number of training epochs (default: 180)
--lr                        initial learning rate (default: 0.1)
--lr-decay                  learning rate decay ratio (default: 0.1)
--lr-decay-epoch            epoch for the learning rate decaying (default: 80, 120)
--weight-decay              weight decay value (default: 5e-4)
--batch-norm                do we need batch norm in ResNet or not (default: True)
--device                    do we use gpu or not (default: 'gpu')
--seed                      used to reproduce the results (default: 0)
```




ContinuousNet provides some some extras (todo):
```
python3 cli.py [--model] [--scheme] [--n_time_steps_per] [--initial_time_d] [--time_epsilon] [--use_skipinit]

standard arguments:
--model
--scheme
--n_time_steps_per
--initial_time_d
--time_epsilon
--use_skipinit
```


After training the model checkpoint is saved in a folder called results.

## Examples and Performance on CIFAR-10

(todo)

```
python train_resnet.py --name cifar10 --epochs 120 --arch ResNet --lr_decay_epoch 30 60 90 --depth_res 20
```

```
python3 cli.py --model ContinuousNet --scheme euler --dataset CIFAR10 --epochs 120  --lr_decay_epoch 30 60 90 --initial_time_d 2
```

```
python3 cli.py --model ContinuousNet --scheme rk4_classic --dataset CIFAR10 --epochs 120  --lr_decay_epoch 30 60 90 --initial_time_d 2 --weight_decay 1e-4
```


| Model           |  Units           | Refined     | Scheme      | #parms  | Test Accuracy | Time |
| ----------------|:----------------:|:----------: |:----------: |:-------:|:-------------:|:----:|
| ResNet-20 (v2)  | 2-2-2            |  -          | -           | 0.27M   | 91.31%        |48 (m)|
| ContinuousNet   | 2-2-2            |  -          | Euler       | 0.27M   | 91.41%        |20 (m)|
| ContinuousNet   | 2-2-2            |  -          | RK4-classic | 0.27M   | 91.01%        |50 (m)|
| ContinuousNet   | 1-1-1 -> 2-2-2   | 25          | RK4-classic | 0.27M   | 91.09%        |47 (m)|
| ContinuousNet   | 1-1-1 -> 2-2-2   | 25          | Midpoint    | 0.27M   | 90.67%        |29 (m)|



| Model           |  Units          |Refined     | Scheme      | #parms  | Test Accuracy | Time  |
| ----------------|:---------------:|:----------:|:----------: |:-------:|:-------------:|:-----:|
| ResNet-52 (v2)  | 8-8-8           | -          | -           | 0.85M   | 93.11%        |105 (m)|
| ContinuousNet   | 8-8-8           | -          | Euler       | 0.86M   | 93.11%        | 83 (m)|
| ContinuousNet   | 8-8-8           | -          | RK4-classic | 0.86M   | 93.29%        |279 (m)|
| ContinuousNet   | 1-1-1 -> 8-8-8  | 30, 50, 70 | RK4-classic | 0.86M   | 93.06%        |199 (m)|



## Examples and Performance on CIFAR-100

(todo)

| Model             |  Units  | Scheme      | #parms  | Test Accuracy | Time  |
| ------------------|:-------:|:----------: |:-------:|:-------------:|:-----:|
| WideResNet-58     | 8-8-8   | -           |  13.63M |   79.16%      |193 (m)|
| WideContinuousNet | 8-8-8   | Euler       |  13.63M |   78.45%      |159 (m)|



## Evaluation

The training script outputs models to Python pickles. The script,
```
python3 eval_manifestation.py
```
performs a "convergence test" for given models by altering their integrator settings to illustrate the manifestation invariance property.

## License

This implementation is released under the GPL 3, as per LICENSE.

## Additional References

There is a video recording of an acompanying presentation available at [Queiruga, A. F., "Continuous-in-Depth Neural Networks," 1st Workshop on Scientific-Driven Deep Learning, July 1, 2020.](https://www.youtube.com/watch?v=_aX3T1Smg54)

