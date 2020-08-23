# Continuous-in-Depth Neural Networks

<img src="https://raw.githubusercontent.com/afqueiruga/ContinuousNet/head/continuousnet_graph_manifestation.jpg" height=600px></img>

This directory contains the implementation of ContinuousNet to accompany the preprint
> [Queiruga, Alejandro F., N. Benjamin Erichson, Dane Taylor and Michael W. Mahoney. “Continuous-in-Depth Neural Networks.” ArXiv abs/2008.02389 (2020)](https://arxiv.org/abs/2008.02389)

## Requirements

This implementation is based on [PyTorch 1.6](https://pytorch.org/). Further, it requires a fork `torchdiffeq` with slight modifications to expose additional solver options: 
```
pip install git+https://github.com/afqueiruga/torchdiffeq
```

## Training

The file `script_it.py` contains configurations to generate the results presented in the main text.
```
python3 script_it.py
```
The training routine can also be executed via `cli.py`. Both are wrappers to `driver.py`, which in turn calls `continuous_net/refine_train.py`.

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

