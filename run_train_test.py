"""Yet another test that runs a tiny model for one epoch on real data."""
from continuous_net_jax.run_experiment import *


DIR = "../runs_cifar10_b_test/"


if __name__=='__main__':
    run_an_experiment(
        dataset_name = 'CIFAR10',
        save_dir=DIR,
        which_model="Continuous",
        alpha=8, hidden=8, n_step=1, n_basis=1, norm="BatchNorm",
        basis='piecewise_constant',
        scheme='Euler',
        kernel_init='kaiming_out',
        n_epoch=5,
        learning_rate=0.1, learning_rate_decay=0.1,
        learning_rate_decay_epochs=[90, 110,],
        refine_epochs=[1,2,3])

