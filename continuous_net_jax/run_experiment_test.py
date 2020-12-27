import tempfile
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
from jax.config import config
config.enable_omnistaging()

from continuous_net_jax.run_experiment import *


class RunExperimentTests(unittest.TestCase):

    def setUp(self):
        self.train_data = [
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
        ]
        self.test_data = [
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
        ]

    def testNoState(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_an_experiment(self.train_data,
                              self.test_data,
                              tmp,
                              alpha=1,
                              hidden=1,
                              n_step=1,
                              n_basis=1,
                              n_epoch=1,
                              norm='None')

    def testStateful(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_an_experiment(self.train_data,
                              self.test_data,
                              tmp,
                              alpha=1,
                              hidden=1,
                              n_step=1,
                              n_basis=1,
                              n_epoch=1,
                              norm='BatchNorm')
