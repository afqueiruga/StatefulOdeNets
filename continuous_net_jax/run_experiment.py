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
from .baselines import ResNet
from .learning_rate_schedule import LearningRateSchedule
from .optimizer_factory import make_optimizer
from .tensorboard_writer import TensorboardWriter
from .training import Trainer, Tester
from .tools import count_parameters


def run_an_experiment(train_data,
                      test_data,
                      save_dir: str = './runs',
                      which_model: str = 'Continuous',
                      seed: int = 0,
                      alpha: int = 8,
                      hidden: int = 8,
                      n_step: int = 3,
                      n_basis: int = 3,
                      scheme: str = 'Euler',
                      norm: str = 'None',
                      kernel_init: str = 'kaiming_out',
                      which_optimizer: str = 'Momentum',
                      learning_rate: float = 0.1,
                      learning_rate_decay: float = 0.1,
                      learning_rate_decay_epochs: Optional[List[int]] = None,
                      weight_decay: float = 5.0e-4,
                      n_epoch: int = 15):
    lr_schedule = LearningRateSchedule(learning_rate, learning_rate_decay,
                                       learning_rate_decay_epochs)
    optimizer_def = make_optimizer(which_optimizer,
                                   learning_rate=learning_rate,
                                   weight_decay=weight_decay)

    if which_model == 'Continuous':
        model = ContinuousImageClassifier(alpha=alpha,
                                              hidden=hidden,
                                              n_step=n_step,
                                              scheme=scheme,
                                              n_basis=n_basis,
                                              norm=norm)
    elif which_model == 'ResNet':
        model = ResNet(alpha=alpha,
                           hidden=hidden,
                           n_step=n_step,
                           norm=norm,
                           kernel_init=kernel_init)
    else:
        raise ArgumentError("Unknown model class.")
    eval_model = model.clone(training=False)

    exp = Experiment(model, path=save_dir)
    exp.save_optimizer_hyper_params(optimizer_def, seed)
    tb_writer = TensorboardWriter(exp.path)
    loss_writer = tb_writer.Writer('loss')
    test_acc_writer = tb_writer.Writer('test_accuracy')
    train_acc_writer = tb_writer.Writer('train_accuracy')

    prng_key = jax.random.PRNGKey(seed)
    x, _ = next(iter(train_data))
    init_vars = exp.model.init(prng_key, x)
    init_state, init_params = init_vars.pop('params')
    optimizer = optimizer_def.create(init_params)
    current_state = init_state
    print("Model has ", count_parameters(init_params), " params + ",
          count_parameters(init_state), " state params (",
          count_parameters(init_vars), " total).")
    trainer = Trainer(exp.model, train_data)
    tester = Tester(eval_model, test_data)

    # print("current_state: ", current_state)
    test_acc = tester.metrics_over_test_set(optimizer.target, current_state)
    test_acc_writer(float(test_acc))
    print("Initial acc ", test_acc)
    for epoch in range(1, 1 + n_epoch):
        optimizer, current_state = trainer.train_epoch(optimizer, current_state,
                                                       lr_schedule(epoch),
                                                       loss_writer,
                                                       train_acc_writer)
        test_acc = tester.metrics_over_test_set(optimizer.target,
                                                 current_state)
        test_acc_writer(float(test_acc))
        print("After epoch ", epoch, " acc: ", test_acc)
        if epoch % 20 == 0:
            exp.save_checkpoint(optimizer, current_state, epoch)
        tb_writer.flush()
    try:
        exp.save_checkpoint(optimizer, current_state, epoch)
    except:
        pass
