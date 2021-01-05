import numpy as onp
import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from . import nonauto_ode_solvers, stateful_ode_solvers
from .nonauto_ode_solvers import OdeIntegrateFast
from .stateful_ode_solvers import StateOdeIntegrateFast

from .basis_functions import *
from .residual_modules import ShallowNet, ResidualUnit, ResidualStitch


# Initialize tools
def copy_and_perturb(params, n_basis):
    """Make copies from a residual trace to make one per coefficient."""
    prng_key = jax.random.PRNGKey(0)
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)

    def _map(x, key):
        return jnp.array(x) + 0.1 * jax.random.normal(key, shape=x.shape)

    return [jax.tree_map(lambda x: _map(x, k), params) for k in subkeys]


# deprecated
def initialize_multiple_times(prng_key, module, x, n_basis):
    """Initilize module on x multiple times by splitting prng_key."""
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)
    return [module.init(k, x) for k in subkeys]


def initialize_multiple_times_split_state(prng_key, module, x, n_basis):
    """Initilize module on x multiple times by splitting prng_key."""
    key, *subkeys = jax.random.split(prng_key, 1 + n_basis)
    params = []
    states = []
    for i, k in enumerate(subkeys):
        inits = module.init(key, x)
        state_i, p = inits.pop('params')
        states.append(state_i)
        param_i = {'params': p}
        params.append(param_i)
    return params, states


def zip_time_dicts(params, states):
    zipped = []
    for i in range(len(params)):
        zipped.append({**params[i], **states[i]})
    return zipped


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
    scheme: str = 'Euler'
    n_basis: int = 1
    basis: BasisFunction = piecewise_constant
    training: bool = True

    def make_param_nodes(self, key, x):
        return initialize_multiple_times_split_state(key, self.R, x,
                                                     self.n_basis)[0]

    def make_state_nodes(self, x):
        key = jax.random.PRNGKey(0)
        return initialize_multiple_times_split_state(key, self.R, x,
                                                     self.n_basis)[1]

    @nn.compact
    def __call__(self, x):
        ode_params = self.param('ode_params', self.make_param_nodes, x)
        ode_states = self.variable('ode_state', 'state', self.make_state_nodes,
                                   x)
        if ode_states.value[0].keys():
            full_params = zip_time_dicts(ode_params, ode_states.value)
            params_of_t = piecewise_constant(full_params)
            r = lambda t, x: self.R.apply(
                params_of_t(t), x, mutable=ode_states.value[0].keys())
            y, t_points, state_points = StateOdeIntegrateFast(
                r, x, scheme=self.scheme, n_step=self.n_step)
            new_state = point_project_tree(state_points, t_points, self.n_basis,
                                           self.basis)
            if self.training:
                ode_states.value = new_state
        else:
            params_of_t = piecewise_constant(ode_params)
            r = lambda t, x: self.R.apply(params_of_t(t), x)
            y = OdeIntegrateFast(r, x, scheme=self.scheme, n_step=self.n_step)
        return y

