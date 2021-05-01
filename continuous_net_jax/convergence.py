from typing import Tuple

from SimDataDB import SimDataDB2
import glob
import os

import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp

from continuous_net_jax import *
from continuous_net import datasets


class ConvergenceTester:

    def __init__(self, path: str):
        self.path = path

        exp = Experiment(path=path, scope=globals())
        # The model was saved at the begining, got longer after refinement.
        final_n_step = exp.model.n_step * 2**len(exp.extra['refine_epochs'])
        final_model = exp.model.clone(n_step=final_n_step)
        # Load the parameters
        chp = checkpoints.restore_checkpoint(path, None)
        params = chp['optimizer']['target']
        state = chp['state']
        # Initialize a skeleton with the right shape.
        prng_key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 32, 32, 3), jnp.float32)
        p = final_model.init(prng_key, x)
        i_state, i_params = p.pop('params')
        # Reshape the values that were loaded. This is needed because
        # ContinuousNet uses lists in the parameter trees, but the
        # checkpoint always loads dictionaries. I.e., we turn
        # {'ContinuousNet0':{'0':W0, '1':W1}} into
        # {'ContinuousNet0':[W0, W1]}
        loaded_params = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(i_params),
            jax.tree_util.tree_leaves(chp['optimizer']['target']))
        loaded_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(i_state),
            jax.tree_util.tree_leaves(chp['state']))
        eval_model = final_model.clone(training=False)

        self.exp = exp
        self.params = loaded_params
        self.state = loaded_state
        self.eval_model = eval_model

    def perform_convergence_test(self,
                                 test_data: Any,
                                 n_steps: Iterable[int],
                                 schemes: Iterable[str]):

        @SimDataDB2(os.path.join(self.path, "convergence.sqlite"))
        def infer_test_error(scheme: str, n_step: int) -> Tuple[float]:
            model = self.eval_model.clone(n_step=n_step)
            tester = Tester(model, test_data)
            err = tester.metrics_over_test_set(self.params, self.state)
            return float(err),

        errors = []
        for n_step in n_steps:
            for scheme in schemes:
                error = infer_test_error(scheme, n_step)
                errors.append((n_step, error))
        return errors
