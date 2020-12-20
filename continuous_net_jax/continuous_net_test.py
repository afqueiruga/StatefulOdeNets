import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

from .continuous_net import *
from .residual_modules import ResidualUnit


class MyMod(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)


class InitializeHelperTests(unittest.TestCase):
    def testInitialize(self):
        prng_key = jax.random.PRNGKey(0)
        x = jnp.array([1.0])
        n_basis = 2
        params = initialize_multiple_times(prng_key, MyMod(), x, n_basis)
        self.assertEqual(len(params), n_basis)


class ContinuousNetTests(unittest.TestCase):
    def testInitNoState(self):
        pass
    def testInitState(self):
        prng_key = jax.random.PRNGKey(0)
        x = jnp.ones((1,12,12,1))
        model = ContinuousNet(ResidualUnit(3), n_basis=2)
        var = model.init(prng_key, x)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
