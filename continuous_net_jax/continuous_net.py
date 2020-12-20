import numpy as onp
import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from .nonauto_ode_solvers import *
from .residual_modules import ShallowNet, ResidualUnit, ResidualStitch


# Initialize tools
def copy_and_perturb(params, n_basis):
    """Make copies from a residual trace to make one per coefficient."""
    prng_key = jax.random.PRNGKey(0)
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)

    def _map(x, key):
        return jnp.array(x) + 0.1 * jax.random.normal(key, shape=x.shape)

    return [jax.tree_map(lambda x: _map(x, k), params) for k in subkeys]


def initialize_multiple_times(prng_key, module, x, n_basis):
    """Initilize module on x multiple times by splitting prng_key."""
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)
    return [module.init(k, x) for k in subkeys]


class ContinuousNet(nn.Module):
    """A continuously deep network block, aka "OdeBlock".

    It obeys the equation:
      dh/dt = R(theta(t), h(t))
    where
      theta(t) = sum phi^a(t) theta^a for a=(0, n_basis)
    With scheme=Euler and basis=piecewise_constant, this network degenerates
    into a residual network.

    Attributes:
      R: the module to use as the rate equation.
      #ode_dim: how many dimensions is the hidden continua.
      #hidden_dim: how many dimensions inside of R_module.
      n_step: how many time steps.
      basis: what basis function is theta?
      n_basis: how many basis function nodes are initialized?
    """
    R: nn.Module
    n_step: int = 1
    scheme: IntegrationScheme = Euler
    n_basis: int = None  # Only needed by init
    basis: BasisFunction = piecewise_constant

    def make_param_nodes(self, key, x):
        # p = self.R.init(key, x)
        # return copy_and_perturb(p, self.n_basis)
        return initialize_multiple_times(key, self.R, x, self.n_basis)

    @nn.compact
    def __call__(self, x):
        ode_params = self.param('ode_params', self.make_param_nodes, x)
        params_of_t_ = params_of_t(ode_params, piecewise_constant)
        return OdeIntegrateFast(params_of_t_, x, self.R.apply,
                                scheme=self.scheme, n_step=self.n_step)

