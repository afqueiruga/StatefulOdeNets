from typing import Tuple
import sys

from continuous_net_jax.run_experiment import *

# import jax.profiler


root = './'
DIR = "../runs_cifar10_b/"

# Baseline ResNet
# run_an_experiment(
#     train_data, test_data, DIR,
#     which_model="ResNet",
#     alpha=16, hidden=16, n_step=16, norm="BatchNorm",
#     kernel_init='kaiming_out',
#     n_epoch=3,
#     learning_rate=0.01, learning_rate_decay=0.1,
#     learning_rate_decay_epochs=[90, 110,],
#     refine_epochs=[])

# Grow to 1 to 16
for SCHEME in ['Euler', 'RK4']:
    run_an_experiment(
          dataset_name='CIFAR10',
          save_dir=DIR,
          which_model="Continuous",
          alpha=16, hidden=16, n_step=1, n_basis=1, norm="BatchNorm",
          basis='piecewise_constant',
          scheme=SCHEME,
          kernel_init='kaiming_out',
          n_epoch=120,
          learning_rate=0.1, learning_rate_decay=0.1,
          learning_rate_decay_epochs=[90, 110,],
          refine_epochs=[20,40,60,80,])
