from typing import Tuple
import sys

from continuous_net_jax.run_experiment import *

import argparse


# import jax.profiler

# export CUDA_VISIBLE_DEVICES=0; python run_cifar10_jax_version.py --seed 0


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--wd', type=float, default=5e-4, help='weight decay parameter')
parser.add_argument('--seed', type=int, default=1, help='seed')

args = parser.parse_args()
print(args)

root = './'
DIR = "../runs_cifar10_b/"

# Experiment 1 1-1-1 ResNet-12
for SCHEME in ['Euler']:
    run_an_experiment(
          dataset_name='CIFAR10',
          save_dir=DIR,
          which_model="Continuous2",
          alpha=1, hidden=16, n_step=2, n_basis=2, 
          norm="BatchNorm-opt-flax",
          basis='piecewise_constant',
          scheme=SCHEME,
          kernel_init='kaiming_out',
          n_epoch=160,
          learning_rate=0.1, learning_rate_decay=0.1, weight_decay=args.wd,
          learning_rate_decay_epochs=[80, 120, 150],
          refine_epochs=[],
          seed=args.seed)
