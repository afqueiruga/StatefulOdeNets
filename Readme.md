# RefineNet: A Continuous-in-Depth Neural Network
## Supplementary Material

This directory contains the implementation of RefineNet.

## Requirements

This implementation is based on PyTorch.
It requires a fork `torchdiffeq` with slight modifications to expose additional solver options. The patched version is included in this supplementary material for anonymity. (Pull requests will be submitted after paper review.)

## Training

The file `script_it.py` contains configurations to generate the results presented in the main text.
```
python3 script_it.py
```
The training routine can also be executed via `cli.py`. Both are wrappers to `driver.py`, which in turn calls `refine_net/refine_train.py`.

## Evaluation

The training script outputs models to Python pickles. The script,
```
python3 eval_manifestation.py
```
produces the plots shown by Figure 4 in the main text, wherein a "convergence test" is performed for given models by altering their integrator settings to illustrate the manifestation invariance property.

