import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp

from .continuous_models import *


class ContinuousImageClassifierTests(unittest.TestCase):

    def setUp(self):
        self.prng_key = jax.random.PRNGKey(0)
        self.x = jnp.ones((1,32,32,3))
        
    def testInitializeStateful(self):
        model = ContinuousImageClassifier()
        params = model.init(self.prng_key, self.x)

    def testInitializeNoState(self):
        model = ContinuousImageClassifier(norm="None")
        params = model.init(self.prng_key, self.x)

    def testApplyStateful(self):
        model = ContinuousImageClassifier()
        theta = model.init(self.prng_key, self.x)
        state, params = theta.pop('params')
        y, out_state = model.apply(theta, self.x, mutable=state.keys())
        self.assertEqual(y.shape, (1, 10))
        self.assertEqual(out_state.keys(), state.keys())

    def testApplyNoState(self):
        model = ContinuousImageClassifier(norm="None")
        params = model.init(self.prng_key, self.x)
        y = model.apply(params, self.x)
        self.assertEqual(y.shape, (1, 10))
