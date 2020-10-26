import numpy as np
import jax

import flax

from nonauto_ode_solver import Euler
from residual_modules import ShallowNet

def SimpleContinuousNet(params, x, ode_dim, h_dim, o_dim, n_step=10, scheme=Euler):
    """Toy version of a continuous net.

    This function is agnostic to the number of basis functions: it is 
    discovered by n_basis = len(params['ode])."""
    # First, linear transform up to the ode dimenions,
    h = flax.nn.Dense.call(params['front'], x, features=ode_dim, bias=False)
    # Create the two functions:
    params_of_t = params_of_t_(params['ode'])  # theta(t)
    f = jax.partial(ShallowNet.call, h_dim=h_dim, o_dim=ode_dim)  # f(theta, x)
    # Now loop over depth-time. This happens **statically**
    dt = 1.0/n_step
    for t in onp.linspace(0, 1, n_step):
        # A 'ResNet' is just this...
        # h = h + dt*ShallowNet.call(params_of_t(t), h, h_dim, h_dim)
        # But the general famility is this: (note how params_of_t is a function)
        h = scheme(params_of_t, h, t, f, dt)
    # Now do a linear classifier on the features of h(t=1)
    y = flax.nn.Dense.call(params['back'], h, features=o_dim, bias=False)
    return jax.nn.sigmoid(y)
