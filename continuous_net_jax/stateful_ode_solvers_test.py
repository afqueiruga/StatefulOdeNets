import unittest

from .stateful_ode_solvers import *

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()


def stateful_f(p, x):
    return x + p + 1.0, p + 1.0


def wrap_f(f_module, params_of_t):
    return lambda t, x: f_module(params_of_t(t), x)


class FlaxMod(nn.Module):

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(1)(x)
        h = nn.BatchNorm()(h)
        return nn.Dense(x.shape[-1])(x)


class StatefulIntegratorsTests(unittest.TestCase):

    def test_euler(self):
        params_of_t = lambda t: 2.0
        f = wrap_f(stateful_f, params_of_t)
        x_out, ts, state = Euler(f, 3.0, 0.0, 1.0)
        self.assertEqual(ts, (0.0,))
        self.assertEqual(state, (3.0,))
        self.assertEqual(x_out, 9.0)

    def test_midpoint(self):
        params_of_t = lambda t: 0.0
        f = wrap_f(stateful_f, params_of_t)
        x_out, ts, state = Midpoint(f, 1.0, 0.0, 1.0)
        self.assertEqual(ts, (0.0, 0.5))
        self.assertEqual(state, (1.0, 1.0))

        self.assertEqual(x_out, 4.0)  # 1.0 + 1.0*( 1.0 + 0.5*(1.0+1.0) + 1.0)

    def test_flax_module(self):
        prng_key = jax.random.PRNGKey(0)
        x = jnp.array([1.0])
        params = FlaxMod().init(prng_key, x)
        init_state, train_params = params.pop('params')

        params_of_t = lambda t: params
        f = lambda t, x: FlaxMod().apply(
            params_of_t(t), x, mutable=init_state.keys())
        x_out, ts, state_out = Euler(f, x, t0=0.0, Dt=1.0)
        self.assertEqual(ts[0], 0.0)
        self.assertEqual(state_out[0].keys(), init_state.keys())

    def test_integration_simple(self):
        params_of_t = lambda t: 2.0
        f = wrap_f(stateful_f, params_of_t)
        x_out, ts, states = StateOdeIntegrateFast(f, 0.0, Euler, 10)
        self.assertEqual(len(states), 10)
        for i, (t_x, state) in enumerate(zip(ts, states)):
            self.assertAlmostEqual(t_x, float(i) / 10)
            self.assertEqual(state, 3.0)

    def test_integration_flax(self):
        prng_key = jax.random.PRNGKey(0)
        x = jnp.array([[1.0]])
        params = FlaxMod().init(prng_key, x)
        init_state, train_params = params.pop('params')

        params_of_t = lambda t: params
        f = lambda t, x: FlaxMod().apply(
            params_of_t(t), x, mutable=init_state.keys())
        x_out, ts, states = StateOdeIntegrateFast(f, x, Euler, 10)
        self.assertEqual(len(states), 10)

        for i, (t_x, state) in enumerate(zip(ts, states)):
            self.assertAlmostEqual(t_x, float(i) / 10)
            self.assertEqual(state.keys(), init_state.keys())


if __name__ == "__main__":
    unittest.main()
