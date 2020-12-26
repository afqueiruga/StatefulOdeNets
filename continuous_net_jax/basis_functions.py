import jax
import jax.numpy as jnp

from .continuous_types import *


# Basis functions
def piecewise_constant(param_nodes: Iterable[JaxTreeType]) -> ContinuousParameters:
    """A piecewise constant basis set.

    Returns:
      A closure on the parameters and bases as function of t.
    """
    n_basis = len(param_nodes)
    def theta(t: float) -> JaxTreeType:
        idx = min(int(n_basis * t), n_basis - 1)
        return param_nodes[idx]
    return theta


def hessian(f):
    return jax.jacfwd(jax.jacrev(f))


def point_loss(params, basis, ts, ys):
    losses = []
    for t_i, y_i in zip(ts, ys):
        loss = (basis(params)(t_i) - y_i)**2
        losses.append(loss)
    return jnp.sum(jnp.array(losses))


def point_project(ys, ts, n_basis, basis):
    params = jnp.zeros(n_basis)
    vG = jax.grad(point_loss)(params, basis, ts, ys)
    mH = hessian(point_loss)(params, basis, ts, ys)
    d_params = -jnp.linalg.solve(mH, vG)
    return d_params


def point_project_tree(tree_point_cloud, ts, n_basis, basis):
    def point_project_list(*args):
        return list(point_project(args, ts, n_basis, basis))
    out = jax.tree_multimap(point_project_list, *tree_point_cloud)
    original_struct = jax.tree_structure(tree_point_cloud[0])
    mapped_struct = jax.tree_structure(list(range(n_basis)))
    return jax.tree_transpose(original_struct, mapped_struct, out)

