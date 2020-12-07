from typing import Tuple

from SimDataDB import SimDataDB2
import glob
import os

import flax
import flax.linen as nn
from flax import optim
import jax
import jax.numpy as jnp

from continuous_net_jax import *
from continuous_net import datasets


def load_for_test(path):
    exp = Experiment(path=path, scope=globals())

    # For now, we to make a skeleton optimizer to interperet the checkpoint.
    prng_key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 28, 28, 1), jnp.float32)
    ode_params = exp.model.init(prng_key, x)['params']
    optimizer_def = optim.Adam(learning_rate=0.001)
    optimizer = optimizer_def.create(ode_params)
    # Load the final checkpoint, using the dict structure.
    optimizer = exp.load_checkpoint(optimizer)
    params = optimizer.target

    # TODO Get this to work:
    # dd = exp.load_checkpoint()
    # params = flax.core.FrozenDict(dd['target'])
    return exp, params

def perform_convergence_test(exp, params, train_data, test_data):
    @SimDataDB2(os.path.join(exp.path, "convergence.sqlite"), "convergence")
    def infer_test_error(scheme: str, n_step: int) -> Tuple[float]:
        model = exp.model.clone(n_step=n_step)
        trainer = Trainer(model, train_data, test_data)
        err = trainer.metrics_over_test_set(params)
        return float(err),
    errs = []
    for n_step in range(1,10):
        err = infer_test_error("Euler", n_step)
        errs.append((n_step, err))
    return errs

def perfom_tests_for_path(path: str, train_data, test_data):
    exp, params = load_for_test(path)
    errors = perform_convergence_test(exp, params, train_data, test_data)
    return exp, errors
