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
import tensorflow.summary as tf_summary
import tqdm

jax.config.enable_omnistaging()

from continuous_net import datasets
from continuous_net_jax import *
from continuous_net_jax.baselines import ResNet


def run_an_experiment(train_data,
                        test_data,
                        save_dir: str = './runs',
                        seed: int = 0,
                        alpha: int = 8,
                        hidden: int = 8,
                        n_step: int = 3,
                        n_basis: int = 3,
                        scheme: str = 'Euler',
                        learning_rate: float = 0.001):
    model = ContinuousImageClassifer(alpha=alpha,
                                     hidden=hidden,
                                     n_step=n_step,
                                     n_basis=n_basis,
                                     scheme=scheme)

    exp = Experiment(model, path=save_dir)
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    exp.save_optimizer_hyper_params(optimizer_def, seed)
    summary_writer = tf_summary.create_file_writer(exp.path)

    prng_key = jax.random.PRNGKey(seed)
    x = jnp.ones((1, 28, 28, 1), jnp.float32)
    ode_params = exp.model.init(prng_key, x)['params']
    optimizer = optimizer_def.create(ode_params)
    trainer = Trainer(exp.model, train_data, test_data)

    loss_int = 0

    def loss_saver(step, val):
        nonlocal loss_int
        with summary_writer.as_default():
            tf_summary.scalar('loss', val, step=loss_int)
        loss_int += 1

    test_accs = [trainer.metrics_over_test_set(optimizer.target)]
    with summary_writer.as_default():
        tf_summary.scalar('accuracy', test_accs[-1], step=0)
    for epoch in range(1, 16):
        print("Working on epoch ", epoch)
        optimizer = trainer.train_epoch(optimizer, loss_saver)
        test_accs.append(trainer.metrics_over_test_set(optimizer.target))
        with summary_writer.as_default():
            tf_summary.scalar('accuracy', test_accs[-1], step=epoch)
        exp.save_checkpoint(optimizer, epoch)
    tf_summary.flush()
    plt.plot([l for l in test_accs], '-o')
    plt.show()
