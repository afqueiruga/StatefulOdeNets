import jax
import jax.numpy as jnp

from .continuous_types import *


# Basis functions
def piecewise_constant(
        param_nodes: Iterable[JaxTreeType]) -> ContinuousParameters:
    """A piecewise constant basis set.

    Returns:
      theta(t) A closure on the parameters and bases as function of t.
    """
    n_basis = len(param_nodes)

    def theta(t: float) -> JaxTreeType:
        idx = min(int(n_basis * t), n_basis - 1)
        return param_nodes[idx]

    return theta


def fem_linear(param_nodes: Iterable[JaxTreeType]) -> ContinuousParameters:
    """Finite Element Method style linear basis functions.
    
    Requires n_basis > 2."""
    n_basis = len(param_nodes)
    n_elem = n_basis - 1

    def theta(t: float) -> JaxTreeType:
        elem_idx = min(int(n_elem * t), n_elem - 1)
        phi_1 = (t - elem_idx / n_elem) / (1.0 / n_elem)
        phi_2 = 1.0 - phi_1
        return jax.tree_multimap(lambda a1, a2: a1 * phi_1 + a2 * phi_2,
                                 param_nodes[elem_idx + 1],
                                 param_nodes[elem_idx])

    return theta


def poly_linear(param_nodes: Iterable[JaxTreeType]) -> ContinuousParameters:
    """Linear polynomial basis functions.
    
    Rolled out, a time dependent layer of this form is:
      y = [W + C*t] x + (b + c*t)
    For each tensor, the function is
      theta(t) = A0 + A1*t
    where A0 and A1 have the same shape, but different scale behavior.

    Requires n_basis = 2."""
    assert len(param_nodes) == 2

    def theta(t: float) -> JaxTreeType:
        return jax.tree_multimap(lambda a, c: a + c * t, param_nodes[0],
                                 param_nodes[1])

    return theta


BASIS = {
    'piecewise_constant': piecewise_constant,
    'fem_linear': fem_linear,
    'poly_linear': poly_linear,
}


def split_refine_piecewise(nodes: Iterable[JaxTreeType]):
    new_nodes = []
    for node in nodes:
        new_nodes.append(node)
        new_nodes.append(node)
    return new_nodes


def split_refine_fem(nodes: Iterable[JaxTreeType]):
    # 
    new_nodes = [nodes[0]]
    for i in range(len(nodes)-1):
        new_nodes.append(0.5*nodes[i]+0.5*nodes[i+1])
        new_nodes.append(nodes[i+1])
    return new_nodes


REFINE = {
    'piecewise_constant': split_refine_piecewise,
    'fem_linear': split_refine_fem,
}


def point_loss(params, basis, ts, ys):
    losses = []
    for t_i, y_i in zip(ts, ys):
        loss = (basis(params)(t_i) - y_i)**2
        losses.append(loss)
    return jnp.sum(jnp.array(losses))


def point_project(ys, ts, n_basis, basis):
    params = jnp.zeros(n_basis)
    vG = jax.grad(point_loss)(params, basis, ts, ys)
    mH = jax.hessian(point_loss)(params, basis, ts, ys)
    d_params = -jnp.linalg.solve(mH, vG)
    return d_params


def point_project_array(ys, ts, n_basis, basis):
    point_project_mapped = lambda ys_: point_project(ys_, ts, n_basis, basis)
    ys_stack = jnp.array(ys)
    ys_flat = ys_stack.reshape(ys_stack.shape[0], -1)
    # print(ys_flat.shape)
    nodes = jax.vmap(point_project_mapped, in_axes=-1, out_axes=-1)(ys_flat)
    # print(nodes.shape)
    return list(nodes.reshape((n_basis,) + ys_stack.shape[1:]))


def point_project_tree(tree_point_cloud, ts, n_basis, basis):

    def point_project_list(*args):
        return list(point_project_array(args, ts, n_basis, basis))

    out = jax.tree_multimap(point_project_list, *tree_point_cloud)
    original_struct = jax.tree_structure(tree_point_cloud[0])
    mapped_struct = jax.tree_structure(list(range(n_basis)))
    return jax.tree_transpose(original_struct, mapped_struct, out)


