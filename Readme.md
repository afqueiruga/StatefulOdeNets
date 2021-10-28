# Stateful ODE-Nets

Introduction (todo).



## Get Started

Just clone the ContinuousNet repository to your local system and you are ready to go:
```
git clone https://github.com/erichson/StatefulOdeNets
```


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


| Model           |  N           | Refined     | Scheme      | #parms  | Test Accuracy |
| ----------------|:----------------:|:----------: |:----------: |:-------:|:-------------:|
ContinuousNet     | 16 | - | Euler | 1.63M | 0.9369 


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

