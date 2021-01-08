"""These are application-complete architectures based on continuousnet."""

import flax.linen as nn
import jax.numpy as jnp

from .continuous_types import *
from .continuous_net import ContinuousNet
from .residual_modules import NORMS, ResidualUnit, ResidualStitch, INITS

from .basis_functions import piecewise_constant
from .residual_modules import ShallowNet


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


class ContinuousImageClassifier(nn.Module):
    """Analogue of the 3-block resnet architecture."""
    alpha: int = 8
    hidden: int = 8
    n_classes: int = 10
    n_step: int = 2
    scheme: str = "Euler"
    n_basis: int = 2
    basis: str = 'piecewise_constant'
    norm: str = "BatchNorm"
    kernel_init: str = 'kaiming_out'
    training: bool = True

    @nn.compact
    def __call__(self, x):
        alpha = self.alpha
        hidden = self.hidden
        # Helper macro.
        R_ = lambda hidden_: ResidualUnit(
            hidden_features=hidden_, norm=self.norm, training=self.training)
        # First filter to make features.
        h = nn.Conv(features=alpha,
                    kernel_size=(3, 3),
                    kernel_init=INITS[self.kernel_init])(x)
        # TODO batchnorm + relu here
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = nn.relu(h)
        # 3 stages of continuous segments:
        h = ContinuousNet(R=R_(hidden),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=hidden,
                           output_features=2 * alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training)(h)
        h = ContinuousNet(R=R_(2 * hidden),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=2 * hidden,
                           output_features=4 * alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training)(h)
        h = ContinuousNet(R=R_(4 * hidden),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        # Pool and linearly classify:
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = jnp.mean(h, axis=(1, 2))
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax
