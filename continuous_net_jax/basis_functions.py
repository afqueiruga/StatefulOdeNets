import jax
import jax.numpy as jnp

from .continuous_types import *


# Basis functions
def piecewise_constant(param_nodes: Iterable[JaxTreeType], t: float,
                       n_basis: int) -> JaxTreeType:
    """A piecewise constant basis set."""
    idx = min(int(n_basis * t), n_basis - 1)
    return param_nodes[idx]


def params_of_t(
        param_nodes: Iterable[JaxTreeType],
        basis: BasisFunction = piecewise_constant) -> ContinuousParameters:
    """Creates a closure on the parameters and bases as function of t."""

    def theta(t):
        return basis(param_nodes, t, len(param_nodes))

    return theta


def hessian(f):
    return jax.jacfwd(jax.jacrev(f))


def point_loss(params, basis, ts, ys):
    losses = []
    for t_i, y_i in zip(ts, ys):
        loss = (params_of_t(params, basis=basis)(t_i) - y_i)**2
        losses.append(loss)
    return jnp.sum(jnp.array(losses))


def point_project(ys, ts, n_basis, basis):
    params = jnp.zeros(n_basis)
    vG = jax.grad(point_loss)(params, basis, ts, ys)
    mH = hessian(point_loss)(params, basis, ts, ys)
    # vG = jnp.stack(G)
    # mH = jnp.stack([jnp.stack(row) for row in H])
    d_params = jnp.linalg.solve(mH, vG)
    return d_params
