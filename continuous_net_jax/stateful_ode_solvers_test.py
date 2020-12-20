import unittest

from .stateful_ode_solvers import *

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

def stateful_f(p, x):
    return x + p + 1.0, p + 1.0


class FlaxMod(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(1)(x)
        return nn.BatchNorm()(h)


class StatefulIntegratorsTests(unittest.TestCase):

    def test_euler(self):
        params_of_t = lambda t: 2.0
        x_out, state = Euler(params_of_t, 3.0, 0.0, stateful_f, 1.0)
        self.assertEqual(state, ((0.0, 3.0),))
        self.assertEqual(x_out, 9.0)

    def test_midpoint(self):
        params_of_t = lambda t: 0.0
        x_out, state = Midpoint(params_of_t, 1.0, 0.0, stateful_f, 1.0)

        self.assertEqual(state, ((0.0, 1.0), (0.5, 1.0)))
        self.assertEqual(x_out, 4.0)  # 1.0 + 1.0*( 1.0 + 0.5*(1.0+1.0) + 1.0)

    def test_flax_module(self):
        prng_key = jax.random.PRNGKey(0)
        x = jnp.array([1.0])
        n_basis = 2
        params = FlaxMod().init(prng_key, x)
        init_state, train_params = params.pop('params')
        params_of_t = lambda t: params
 
        f = lambda *args, **kwargs: FlaxMod().apply(*args, mutable=init_state.keys(), **kwargs)
        x_out, state_out = Euler(params_of_t, x, t0=0.0, f=f, Dt=1.0)

        # print(state)
        # print(x_out)
        self.assertEqual(state_out[0][0], 0.0)
        self.assertEqual(state_out[0][1].keys(), init_state.keys())


if __name__ == "__main__":
    unittest.main()
