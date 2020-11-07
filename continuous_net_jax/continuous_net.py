import numpy as onp
import jax
import jax.numpy as jnp

import flax

from .nonauto_ode_solvers import Euler, RK4, params_of_t_
from .residual_modules import ShallowNet


# Initialize tools
def copy_and_perturb(params, n_basis):
    """Make copies from a residual trace to make one per coefficient."""
    prng_key = jax.random.PRNGKey(0)
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)

    def _map(x, key):
        return jnp.array(x) + 0.1 * jax.random.normal(key, shape=x.shape)

    return [jax.tree_map(lambda x: _map(x, k), params) for k in subkeys]


def init_ode_by_shape(prng_key, ode_model, n_basis, shape, *args):
    """Make multiple coefficients from params using a flax model."""
    _, params = ode_model.init_by_shape(prng_key, shape, *args)
    return copy_and_perturb(params, n_basis)


class OdeBlock(flax.nn.Module):

    @classmethod
    def init_by_shape(cls,
                      _rng,
                      input_specs,
                      R,
                      n_basis,
                      *args,
                      name=None,
                      **kwargs):
        R.init_by_shape(_rng,
                        input_specs=input_specs,
                        *args,
                        name=name,
                        **kwargs)
        return {f'ode': copy_and_perturb(params, n_basis)}

    def apply(x, R, basis_set, n_step, scheme, *args, **kwargs):
        params_of_t = params_of_t_(self.params['ode'], basis_set)
        f = jax.partial(R.call, *args, **kwargs)
        dt = 1.0 / n_step
        for t in onp.linspace(0, 1, n_step):
            h = scheme(params_of_t, h, t, f, dt)
        return h


def SimpleContinuousNet(params,
                        x,
                        ode_dim,
                        h_dim,
                        out_dim,
                        n_step=10,
                        scheme=Euler):
    """Toy version of a continuous net.

    This function is agnostic to the number of basis functions: it is 
    discovered by n_basis = len(params['ode])."""
    # First, linear transform up to the ode dimenions,
    h = flax.nn.Dense.call(params['front'], x, features=ode_dim, bias=False)
    # Create the two functions:
    params_of_t = params_of_t_(params['ode'])  # theta(t)
    f = jax.partial(ShallowNet.call, h_dim=h_dim,
                    out_dim=ode_dim)  # f(theta, x)
    # Now loop over depth-time. This happens **statically**
    dt = 1.0 / n_step
    for t in onp.linspace(0, 1, n_step):
        # A 'ResNet' is just this...
        # h = h + dt*ShallowNet.call(params_of_t(t), h, h_dim, h_dim)
        # But the general famility is this: (note how params_of_t is a function)
        h = scheme(params_of_t, h, t, f, dt)
    # Now do a linear classifier on the features of h(t=1)
    y = flax.nn.Dense.call(params['back'], h, features=out_dim, bias=False)
    return jax.nn.sigmoid(y)


def init_SimpleContinuousNet(ode_dim=3,
                             h_dim=1,
                             n_step=9,
                             n_basis=3,
                             scheme=RK4):
    prng_key = jax.random.PRNGKey(0)
    key, *subkeys = jax.random.split(prng_key, 3)
    params = {
        'front':
            flax.nn.Dense.init_by_shape(prng_key, [(2,)],
                                        features=ode_dim,
                                        bias=False)[1],
        'ode':
            init_ode_by_shape(prng_key, ShallowNet, n_basis, [(ode_dim,)],
                              h_dim, ode_dim),
        'back':
            flax.nn.Dense.init_by_shape(prng_key, [(ode_dim,)],
                                        features=1,
                                        bias=False)[1],
    }
