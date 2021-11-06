# Stateful ODE-Nets

This repository provides research code for [Stateful ODE-Nets](https://arxiv.org/pdf/2106.10820.pdf), presented at NeurIPS (2021). 



## Get Started

Just clone the ContinuousNet repository to your local system and you are ready to go:
```
git clone https://github.com/erichson/StatefulOdeNets
```


## Training




## Examples and Performance on CIFAR-10

ContinuousNets can be trained similar to ResNets via a command line interface. Here are two examples. First, we train an ODE-Net without refinemenet training:

```
python3 run_cifar10.py --which_model ContinuousNet --scheme Euler --n_steps 16 --n_basis 16 --epsilon 16 
```

Next, we train an ODE-Net with refinement training:

```
python3 run_cifar10.py --which_model ContinuousNet --scheme Euler --refine_epochs 20 40 70 90
```

The results are summarized in the following table.

| Model           |  N | K  | Refined     | Scheme      | #parameters  | Test Accuracy |
| ----------------|:--:|:--:|:----------: |:----------: |:------------:|:-------------:|
|ContinuousNet (1)   | 16 | 16 | -           | Euler    | 1.63M        | 0.9369       |
|ContinuousNet (2)   | 16 | 6 | 1->2->4->8->16  | Euler | 1.63M        | 0.927       |



## Compression

The script,
```
python3 run_compression.py
```
is compressing a given model, without retraining or revisiting any data. We present results for the second model that was trained with the refinement training scheme. 


| Model           |  N | K  | Scheme      | #parameters  | Test Accuracy |
| ----------------|:--:|:--:|:----------: |:------------:|:-------------:|
|Compressed ContinuousNet (2)| 8 | 16 | Euler       | 0.85M     | 0.927       |
|Compressed ContinuousNet (2)| 8 | 8 | Euler       | 0.85M     | 0.920       |


## License

This implementation is released under the GPL 3, as per LICENSE.

## References

* Continuous-in-Depth Neural Networks: [https://arxiv.org/pdf/2008.02389.pdf](https://arxiv.org/pdf/2008.02389.pdf)
* Stateful ODE-Nets using Basis Function Expansions: [https://arxiv.org/pdf/2106.10820.pdf](https://arxiv.org/pdf/2106.10820.pdf)


