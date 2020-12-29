from typing import List, Optional

from datetime import datetime

import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from flax import optim
import flax.training.checkpoints
import tensorflow.summary as tf_summary
from matplotlib import pylab as plt
import numpy as np

import tqdm

jax.config.enable_omnistaging()

from continuous_net import datasets
from continuous_net_jax import *
from continuous_net_jax.baselines import ResNet
from .learning_rate_schedule import LearningRateSchedule
from .tensorboard_writer import TensorboardWriter
from .optimizer_factory import make_optimizer


def run_an_experiment(train_data,
                      test_data,
                      save_dir: str = './runs',
                      seed: int = 0,
                      alpha: int = 8,
                      hidden: int = 8,
                      n_step: int = 3,
                      n_basis: int = 3,
                      scheme: str = 'Euler',
                      norm: str = 'None',
                      optimizer: str = 'SGD',
                      learning_rate: float = 0.001,
                      learning_rate_decay: float = 0.8,
                      learning_rate_decay_epochs: Optional[List[int]] = None,
                      n_epoch: int = 15):
    lr_schedule = LearningRateSchedule(learning_rate, learning_rate_decay,
                                       learning_rate_decay_epochs)

    model = ContinuousImageClassifier(alpha=alpha,
                                      hidden=hidden,
                                      n_step=n_step,
                                      n_basis=n_basis,
                                      norm=norm,
                                      scheme=scheme)

    exp = Experiment(model, path=save_dir)
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    exp.save_optimizer_hyper_params(optimizer_def, seed)
    tb_writer = TensorboardWriter(exp.path)
    loss_writer = tb_writer.Writer('loss')
    accuracy_writer = tb_writer.Writer('accuracy')

    prng_key = jax.random.PRNGKey(seed)
    x, _ = next(iter(train_data))
    init_vars = exp.model.init(prng_key, x)
    init_state, init_params = init_vars.pop('params')
    optimizer = optimizer_def.create(init_params)
    current_state = init_state
    trainer = Trainer(exp.model, train_data, test_data)

    test_acc = trainer.metrics_over_test_set(optimizer.target, current_state)
    accuracy_writer(float(test_acc))
    print("Initial acc ", test_acc)
    for epoch in range(1, 1 + n_epoch):
        optimizer, current_state = trainer.train_epoch(optimizer, current_state,
                                                       lr_schedule(epoch),
                                                       loss_writer)
        test_acc = trainer.metrics_over_test_set(optimizer.target, current_state)
        accuracy_writer(float(test_acc))
        exp.save_checkpoint(optimizer, current_state, epoch)
        print("After epoch ", epoch, " acc: ", test_acc)
        tf_summary.flush()
