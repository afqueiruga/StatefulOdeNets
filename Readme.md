# Continuous-in-Depth Neural Networks

<img src="https://raw.githubusercontent.com/afqueiruga/ContinuousNet/head/continuous_graph_manifestation.jpg" height=600px></img>

This directory contains the implementation of ContinuousNet to accompany an upcoming paper. (It will be linked right here if you come back in few days!)

## Requirements

This implementation is based on PyTorch. It requires a fork `torchdiffeq` with slight modifications to expose additional solver options: `github.com/afqueiruga/torchdiffeq`

## Training

The file `script_it.py` contains configurations to generate the results presented in the main text.
```
python3 script_it.py
```
The training routine can also be executed via `cli.py`. Both are wrappers to `driver.py`, which in turn calls `continuous_net/continuous_train.py`.

## Evaluation

The training script outputs models to Python pickles. The script,
```
python3 eval_manifestation.py
```
performs a "convergence test" for given models by altering their integrator settings to illustrate the manifestation invariance property.

## License

This implementation is released under the GPL 3, as per LICENSE.

## Reference
[soon]()
