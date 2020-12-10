import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

from .continuous_net import *


class MyMod(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)


class BasisFunctionTests(unittest.TestCase):
    def testInitialize(self):
        prng_key = jax.random.PRNGKey(0)
        x = jnp.array([1.0])
        n_basis = 2
        params = initialize_multiple_times(prng_key, MyMod(), x, n_basis)
        self.assertEqual(len(params), n_basis)


if __name__ == "__main__":
    unittest.main()
