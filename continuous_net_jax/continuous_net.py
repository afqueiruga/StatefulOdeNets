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
        R = self.R_module(hidden_dim=self.hidden_dim, output_dim=self.ode_dim)
        h = nn.Dense(self.ode_dim)(x)
        h = ContinuousNet(R, self.n_step, self.basis, self.n_basis)(h)
        return nn.sigmoid(nn.Dense(1)(h))


class ContinuousImageClassifer(nn.Module):
    """Analogue of the 3-block resnet architecture."""
    alpha: int = 8
    hidden: int = 8
    n_classes: int = 10
    n_step: int = 2
    scheme: str = "Euler"
    n_basis: int = 2

    @nn.compact
    def __call__(self, x):
        alpha = self.alpha
        hidden = self.hidden
        h = nn.Conv(features=alpha, kernel_size=(3, 3))(x)
        h = ContinuousNet(R=ResidualUnit(hidden_features=hidden),
                          scheme=SCHEME_TABLE[self.scheme],
                          n_step=self.n_step,
                          n_basis=self.n_basis)(h)
        h = ResidualStitch(hidden_features=hidden,
                           output_features=2 * alpha,
                           strides=(2, 2))(h)
        h = ContinuousNet(R=ResidualUnit(hidden_features=2 * hidden),
                          scheme=SCHEME_TABLE[self.scheme],
                          n_step=self.n_step,
                          n_basis=self.n_basis)(h)
        h = ResidualStitch(hidden_features=2 * hidden,
                           output_features=4 * alpha,
                           strides=(2, 2))(h)
        h = ContinuousNet(R=ResidualUnit(hidden_features=4 * hidden),
                          scheme=SCHEME_TABLE[self.scheme],
                          n_step=self.n_step,
                          n_basis=self.n_basis)(h)
        h = nn.pooling.avg_pool(h, (h.shape[-3], h.shape[-2]))
        h = h.reshape(h.shape[0], -1)
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax
