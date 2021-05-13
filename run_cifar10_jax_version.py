from typing import Tuple
import sys

from continuous_net_jax.run_experiment import run_an_experiment

import argparse

# import jax.profiler

# export CUDA_VISIBLE_DEVICES=2; python run_cifar10_jax_version_2.py --alpha 1 --seed 1 --basis piecewise_constant --refine_epochs 70 --n_basis 1 --n_steps 2


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--wd', type=float, default=5e-4, help='weight decay parameter')
parser.add_argument('--scheme', type=str, default="Euler", help='numerical integrator scheme')
parser.add_argument('--basis', type=str, default="piecewise_constant", help='basis function')
parser.add_argument('--alpha', type=int, default=1, help='seed')
parser.add_argument('--n_steps', type=int, default=1, help='number of steps')
parser.add_argument('--n_basis', type=int, default=1, help='number of basis functions')
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[80, 120, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--refine_epochs', type=int, nargs='+', default=[], help='Refinement epochs.')
parser.add_argument('--project_epochs', type=int, nargs='+', default=[], help='Project epochs.')
parser.add_argument('--seed', type=int, default=1, help='seed')


args = parser.parse_args()
print(args)
root = './'
DIR = "../runs_cifar10_b/"

run_an_experiment(
          dataset_name='CIFAR10',
          save_dir=DIR,
          which_model="Continuous",
          alpha=args.alpha, 
          hidden=16, 
          n_step=args.n_steps, 
          n_basis=args.n_basis, 
          norm="BatchNorm-opt-flax",
          basis=args.basis,
          scheme=args.scheme,
          kernel_init='kaiming_out',
          n_epoch=160,
          learning_rate=0.1, learning_rate_decay=0.1, weight_decay=args.wd,
          learning_rate_decay_epochs=args.lr_decay_epoch,
          refine_epochs=args.refine_epochs,
          project_epochs=args.project_epochs,

          seed=args.seed)