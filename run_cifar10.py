from typing import Tuple
import sys

from continuous_net.run_experiment import run_an_experiment

import argparse


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--wd', type=float, default=5e-4, help='weight decay parameter')
parser.add_argument('--scheme', type=str, default="Euler", help='numerical integrator scheme')
parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon')
parser.add_argument('--basis', type=str, default="piecewise_constant", help='basis function')
parser.add_argument('--alpha', type=int, default=1, help='widht of network')
parser.add_argument('--n_steps', type=int, default=1, help='number of steps')
parser.add_argument('--n_basis', type=int, default=1, help='number of basis functions')
parser.add_argument('--epochs', type=int, default=180, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[80, 120, 160], help='Decrease learning rate at these epochs.')
parser.add_argument('--refine_epochs', type=int, nargs='+', default=[], help='Refinement epochs.')
parser.add_argument('--project_epochs', type=int, nargs='+', default=[], help='Project epochs.')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--save_dir', type=str, default="../runs_cifar10_b/", help='path to store the models at')
parser.add_argument('--which_model', type=str, default="ContinuousNet", help='which model')


args = parser.parse_args()
print(args)

root = './'

run_an_experiment(
          dataset_name='CIFAR10',
          save_dir=args.save_dir,
          which_model=args.which_model,
          alpha=args.alpha, 
          hidden=16, 
          n_step=args.n_steps, 
          n_basis=args.n_basis, 
          norm="BatchNorm-opt-flax",
          basis=args.basis,
          scheme=args.scheme,
          epsilon=args.epsilon,
          kernel_init='kaiming_out',
          n_epoch=args.epochs,
          learning_rate=args.learning_rate, learning_rate_decay=0.1, weight_decay=args.wd,
          learning_rate_decay_epochs=args.lr_decay_epoch,
          refine_epochs=args.refine_epochs,
          project_epochs=args.project_epochs,
          seed=args.seed)
