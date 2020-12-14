from typing import List

from datetime import datetime

import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from flax import optim
import flax.training.checkpoints
from matplotlib import pylab as plt
import numpy as np

import tqdm

jax.config.enable_omnistaging()

from continuous_net import datasets
from continuous_net_jax import *
from continuous_net_jax.baselines import ResNet
from .tensorboard_writer import *
from .optimizer_factory import *


def run_an_experiment(train_data,
                      test_data,
                      save_dir: str = './runs',
                      seed: int = 0,
                      alpha: int = 8,
                      hidden: int = 8,
                      n_step: int = 3,
                      n_basis: int = 3,
                      scheme: str = 'Euler',
                      optimizer: str = 'SGD',
                      learning_rate: float = 0.001,
                      n_epoch: int = 15):
    model = ContinuousImageClassifer(alpha=alpha,
                                     hidden=hidden,
                                     n_step=n_step,
                                     n_basis=n_basis,
                                     scheme=scheme)

    exp = Experiment(model, path=save_dir)
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    exp.save_optimizer_hyper_params(optimizer_def, seed)
    tb_writer = TensorboardWriter(exp.path)
    loss_saver = tb_writer.Writer('loss')
    accuracy_writer = tb_writer.Writer('accuracy')

    prng_key = jax.random.PRNGKey(seed)
    x, _ = next(iter(train_data))
    ode_params = exp.model.init(prng_key, x)['params']
    optimizer = optimizer_def.create(ode_params)
    trainer = Trainer(exp.model, train_data, test_data)

    test_accs = [trainer.metrics_over_test_set(optimizer.target)]
    accuracy_writer(test_accs[-1])
    for epoch in range(1, 1 + n_epoch):
        print("Working on epoch ", epoch)
        optimizer = trainer.train_epoch(optimizer, loss_saver)
        test_accs.append(trainer.metrics_over_test_set(optimizer.target))
        accuracy_writer(test_accs[-1])
        exp.save_checkpoint(optimizer, epoch)

    tf_summary.flush()
    plt.plot([l for l in test_accs], '-o')
    plt.show()
