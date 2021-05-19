"""These are application-complete architectures based on continuousnet."""

import flax
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Tuple

from .continuous_types import *
from .continuous_net import ContinuousNet
from .residual_modules import NORMS, ResidualUnit, ResidualStitch, INITS

from .basis_functions import piecewise_constant, REFINE
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
    alpha: int = 1
    hidden: int = 16
    n_classes: int = 10
    n_step: int = 2
    scheme: str = "Euler"
    n_basis: int = 2
    basis: str = 'piecewise_constant'
    norm: str = "BatchNorm"
    kernel_init: str = 'kaiming_out'
    training: bool = True
    epsilon: int = 1.0
    stitch_epsilon: int = 1.0
    

    @nn.compact
    def __call__(self, x):

        # Helper macro.
        R_ = lambda hidden_: ResidualUnit(
            hidden_features=hidden_, norm=self.norm, training=self.training, epsilon=self.epsilon)
        # First filter to make features.
        h = nn.Conv(features=self.hidden * self.alpha,
                    use_bias=False,
                    kernel_size=(3, 3),
                    kernel_init=INITS[self.kernel_init])(x)
        h = NORMS[self.norm](use_running_average=not self.training)(h)        
        h = nn.gelu(h)
        # 3 stages of continuous segments:
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features=self.hidden * self.alpha,
                           strides=(1, 1),
                           norm=self.norm,
                           training=self.training,
                           epsilon=self.stitch_epsilon)(h)        
        h = ContinuousNet(R=R_(self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features=2 * self.hidden * self.alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training,
                           epsilon=self.stitch_epsilon)(h)
        h = ContinuousNet(R=R_(2 * self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=2 * self.hidden * self.alpha,
                           output_features=4 * self.hidden * self.alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training,
                           epsilon=self.stitch_epsilon)(h)
        h = ContinuousNet(R=R_(4 * self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        # Pool and linearly classify:
        h = NORMS[self.norm](use_running_average=not self.training)(h)        
        #h = nn.gelu(h)
        h = nn.avg_pool(h, window_shape=(8, 8), strides=(8, 8))
        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax

    def refine(self,
               params: JaxTreeType,
               state: JaxTreeType = None) -> Tuple[JaxTreeType, JaxTreeType]:
        new_model = self.clone(n_step=2 * self.n_step, n_basis=2 * self.n_basis)
        new_params = {}
        for k, v in params.items():
            if 'Continuous' in k:
                new_params[k] = {
                    'ode_params': REFINE[self.basis](v['ode_params'])
                }
            else:
                new_params[k] = v
        new_params = flax.core.frozen_dict.FrozenDict(new_params)

        if not state:
            return new_model, new_params
        else:
            keep_state, ode_state = state.pop('ode_state')
            new_ode_state = {}
            for k, v in ode_state.items():
                new_ode_state[k] = {'state': REFINE[self.basis](v['state'])}
            new_state = flax.core.frozen_dict.FrozenDict({
                **keep_state, 'ode_state': new_ode_state
            })
            return new_model, new_params, new_state




class ContinuousNetReLU(nn.Module):
    """Analogue of the 3-block resnet architecture."""
    alpha: int = 1
    hidden: int = 16
    n_classes: int = 10
    n_step: int = 2
    scheme: str = "Euler"
    n_basis: int = 2
    basis: str = 'piecewise_constant'
    norm: str = "BatchNorm"
    kernel_init: str = 'kaiming_out'
    training: bool = True
    epsilon: int = 1.0
    stitch_epsilon: int = 1.0    

    @nn.compact
    def __call__(self, x):

        # Helper macro.
        R_ = lambda hidden_: ResidualUnit(
            hidden_features=hidden_, norm=self.norm, training=self.training, epsilon=self.epsilon, activation=nn.relu)
        # First filter to make features.
        h = nn.Conv(features=self.hidden * self.alpha, use_bias=False,
                    kernel_size=(3, 3),
                    kernel_init=INITS[self.kernel_init])(x)
        h = NORMS[self.norm](use_running_average=not self.training)(h)        
        h = nn.relu(h)
        # 3 stages of continuous segments:
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features=self.hidden * self.alpha,
                           strides=(1, 1),
                           norm=self.norm,
                           training=self.training,
                           epsilon=self.stitch_epsilon,
                           activation=nn.relu)(h)        
        h = ContinuousNet(R=R_(self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features=2 * self.hidden * self.alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training,
                           epsilon=self.stitch_epsilon, 
                           activation=nn.relu)(h)
        h = ContinuousNet(R=R_(2 * self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=2 * self.hidden * self.alpha,
                           output_features=4 * self.hidden * self.alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training,
                           epsilon=self.stitch_epsilon,
                           activation=nn.relu)(h)
        h = ContinuousNet(R=R_(4 * self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        # Pool and linearly classify:
        h = NORMS[self.norm](use_running_average=not self.training)(h)        
        #h = nn.relu(h)
        h = nn.avg_pool(h, window_shape=(8, 8), strides=(8, 8))
        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax

    def refine(self, params: JaxTreeType, state: JaxTreeType=None
                   ) -> Tuple[JaxTreeType, JaxTreeType]:
        return refine(self, params, state)


class ContinuousImageClassifierSmall(nn.Module):
    """Analogue of the 3-block resnet architecture."""
    alpha: int = 1
    hidden: int = 16
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
        # Helper macro.
        R_ = lambda hidden_: ResidualUnit(
            hidden_features=hidden_, norm=self.norm, training=self.training, activation=nn.gelu)
        # First filter to make features.
        h = nn.Conv(features=self.hidden * self.alpha,
                    use_bias=False,
                    kernel_size=(3, 3),
                    kernel_init=INITS[self.kernel_init])(x)
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = nn.gelu(h)
        # 2 stages of continuous segments:
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features=self.hidden * self.alpha,
                           strides=(1, 1),
                           norm=self.norm,
                           training=self.training,
                           activation=nn.gelu)(h)        
        h = ContinuousNet(R=R_(self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features=self.hidden * self.alpha * 2,
                           strides=(2, 2),
                           norm=self.norm,
                           training=self.training,
                           activation=nn.gelu)(h)
        h = ContinuousNet(R=R_(self.hidden * self.alpha *2),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)

        # Pool and linearly classify:
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = nn.gelu(h)
        h = nn.avg_pool(h, window_shape=(8, 8), strides=(8, 8))
        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax

    def refine(self,
               params: JaxTreeType,
               state: JaxTreeType = None) -> Tuple[JaxTreeType, JaxTreeType]:
        return refine(self, params, state)
        
        
class ContinuousImageClassifierMNIST(nn.Module):
    """Analogue of the 3-block resnet architecture."""
    alpha: int = 1
    hidden: int = 16
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
        # Helper macro.
        R_ = lambda hidden_: ResidualUnit(
            hidden_features=hidden_, norm=self.norm, training=self.training, activation=nn.gelu)
        # First filter to make features.
        h = nn.Conv(features=self.hidden * self.alpha, use_bias=False,
                    kernel_size=(3, 3),
                    kernel_init=INITS[self.kernel_init])(x)
        h = NORMS[self.norm](use_running_average=not self.training)(h)        
        h = nn.gelu(h)
        # 2 stages of continuous segments:
        h = ResidualStitch(hidden_features=self.hidden * self.alpha,
                           output_features= self.hidden * self.alpha,
                           strides=(1, 1),
                           norm=self.norm,
                           training=self.training,
                           activation=nn.gelu)(h)        
        h = ContinuousNet(R=R_(self.hidden * self.alpha),
                          scheme=self.scheme,
                          n_step=self.n_step,
                          n_basis=self.n_basis,
                          basis=self.basis,
                          training=self.training)(h)

        # Pool and linearly classify:
        h = NORMS[self.norm](use_running_average=not self.training)(h)        
        h = nn.gelu(h)
        h = nn.avg_pool(h, window_shape=(8, 8), strides=(8, 8))
        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax

    def refine(self, params: JaxTreeType, state: JaxTreeType=None
                   ) -> Tuple[JaxTreeType, JaxTreeType]:
        return refine(self, params, state)   



def refine(model,
           params: JaxTreeType, state: JaxTreeType = None) -> Tuple[JaxTreeType, JaxTreeType]:
    new_model = model.clone(n_step=2 * model.n_step, n_basis=2 * model.n_basis)
    new_params = {}
    for k, v in params.items():
        if 'Continuous' in k:
            new_params[k] = {'ode_params': REFINE[model.basis](v['ode_params'])}
        else:
            new_params[k] = v
    new_params = flax.core.frozen_dict.FrozenDict(new_params)

    if not state:
        return new_model, new_params
    else:
        keep_state, ode_state = state.pop('ode_state')
        new_ode_state = {}
        for k, v in ode_state.items():
            new_ode_state[k] = {'state': REFINE[model.basis](v['state'])}
        new_state = flax.core.frozen_dict.FrozenDict({
            **keep_state, 'ode_state': new_ode_state
        })
        return new_model, new_params, new_state
