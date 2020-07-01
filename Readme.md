# RefineNet: A Continuous-in-Depth Neural Network
Alejandro F. Queiruga  
N. Ben Erichson  
2019-2020

![refinenet_graph_manifestation.png](A RefineNet manifests as a family of graphs using basis functions)

This directory contains the implementation of RefineNet to accompany an upcoming paper. (It will be linked right here if you come back in few days!)

## Requirements

This implementation is based on PyTorch. It requires a fork `torchdiffeq` with slight modifications to expose additional solver options: `github.com/afqueiruga/torchdiffeq`

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
performs a "convergence test" for given models by altering their integrator settings to illustrate the manifestation invariance property.

## License

This implementation is released under the GPL 3, as per LICENSE.

## Acknowledgements

A. Queiruga initially developed this code by while at Lawrence Berkeley National Lab with support from the U.S. Department of Energy.

N.B. Erichson is supported by U.C. Berkeley and I.C.S.I.
