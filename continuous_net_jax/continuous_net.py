import numpy as onp
import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from .nonauto_ode_solvers import *
from .residual_modules import ShallowNet


# Initialize tools
def copy_and_perturb(params, n_basis):
    """Make copies from a residual trace to make one per coefficient."""
    prng_key = jax.random.PRNGKey(0)
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)

    def _map(x, key):
        return jnp.array(x) + 0.1 * jax.random.normal(key, shape=x.shape)

    return [jax.tree_map(lambda x: _map(x, k), params) for k in subkeys]


class ContinuousNet(nn.Module):
    """A continuously deep network block, aka "OdeBlock".

    It obeys the equation:
      dh/dt = R(theta(t), h(t))
    where
      theta(t) = sum phi^a(t) theta^a for a=(0, n_basis)
    With scheme=Euler and basis=piecewise_constant, this network degenerates
    into a residual network.

    Attributes:
      R_module: the module to use as the rate equation.
      ode_dim: how many dimensions is the hidden continua.
      hidden_dim: how many dimensions inside of R_module.
      n_step: how many time steps.
      basis: what basis function is theta?
      n_basis: how many basis function nodes are initialized?
    """
    R: nn.Module
    n_step: int
    basis: BasisFunction = piecewise_constant
    n_basis: int = None  # Only needed by init

    def make_param_nodes(self, key, x):
        p = self.R.init(key, x)
        return copy_and_perturb(p, self.n_basis)

    @nn.compact
    def __call__(self, x):
        ode_params = self.param('ode_params', self.make_param_nodes, x)
        params_of_t = params_of_t_(ode_params, piecewise_constant)
        #r_of_theta = lambda t_, x_: self.R(params_of_t(t_), x_)
        return OdeIntegrateFast(params_of_t, x, self.R.apply)


class ContinuousClassifier(nn.Module):
    """A basic fully-connected continuously deep classifier.

    It obeys the equation:
      h(0)  = W x
      dh/dt = R(theta(t), h(t))
      y = sigma(D h(1))

    Attributes:
      R_module: the module to use as the rate equation.
      ode_dim: how many dimensions is the hidden continua.
      hidden_dim: how many dimensions inside of R_module.
      n_step: how many time steps.
      basis: what basis function is theta?
      n_basis: how many basis function nodes are initialized?
    """
    R_module: nn.Module = ShallowNet
    ode_dim: int = 3
    hidden_dim: int = 1
    n_step: int = 10
    basis: BasisFunction = piecewise_constant
    n_basis: int = None  # Only needed by init

    # def setup(self):
    #     self.R = self.R_module(hidden_dim=self.hidden_dim,
    #                            output_dim=self.ode_dim)

    @nn.compact
    def __call__(self, x):
        R = self.R_module(hidden_dim=self.hidden_dim,
                               output_dim=self.ode_dim)
        h = nn.Dense(self.ode_dim)(x)
        h = ContinuousNet(R, self.n_step, self.basis, self.n_basis)(h)
        return nn.sigmoid(nn.Dense(1)(h))
