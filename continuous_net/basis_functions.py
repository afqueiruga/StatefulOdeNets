import jax
import jax.numpy as jnp
import numpy as onp

from .continuous_types import *
from .tools import *

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
    if n_basis == 1:
        return piecewise_constant(param_nodes)
    n_elem = n_basis - 1

    def theta(t: float) -> JaxTreeType:
        elem_idx = min(int(n_elem * t), n_elem - 1)
        phi_1 = (t - elem_idx / n_elem) / (1.0 / n_elem)
        phi_2 = 1.0 - phi_1
        return jax.tree_multimap(lambda a1, a2: a1 * phi_1 + a2 * phi_2,
                                 param_nodes[elem_idx + 1],
                                 param_nodes[elem_idx])

    return theta


def piecewise_linear(param_nodes: Iterable[JaxTreeType]) -> ContinuousParameters:
    """Piecewise linear i.e. Discontinuous Galerkin (DG) linear.
    
    Requiries n_basis % 2 == 0
    """
    assert len(param_nodes) % 2 == 0
    n_elem = len(param_nodes) // 2
    def theta(t: float) -> JaxTreeType:
        elem_idx = min(int(n_elem * t), n_elem - 1)
        phi_1 = (t - elem_idx / n_elem) / (1.0 / n_elem)
        phi_2 = 1.0 - phi_1
        return jax.tree_multimap(lambda a1, a2: a1 * phi_1 + a2 * phi_2,
                                 param_nodes[2*elem_idx + 1],
                                 param_nodes[2*elem_idx])
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
    'piecewise_linear': piecewise_linear,
    'poly_linear': poly_linear,
}


#
# Fast split refinement.
#
def split_refine_piecewise(nodes: Iterable[JaxTreeType]):
    new_nodes = []
    for node in nodes:
        new_nodes.append(node)
        new_nodes.append(node)
    return new_nodes


def split_refine_fem(nodes: Iterable[JaxTreeType]):
    """Traditional finite element hat functions."""
    # Fringe case is constant, which turns into one element.
    if len(nodes) == 1:
        return [nodes[0], nodes[0]]

    # Use the midpoints to subdivide elements.
    new_nodes = [nodes[0]]
    for i in range(len(nodes) - 1):
        midpoint = jax.tree_multimap(lambda a1, a2: a1 * 0.5 + a2 * 0.5,
                                     nodes[i + 1], nodes[i])
        new_nodes.append(midpoint)
        new_nodes.append(nodes[i + 1])
    return new_nodes


REFINE = {
    'piecewise_constant': split_refine_piecewise,
    'fem_linear': split_refine_fem,
}


#
# General Interpolation
#
def piecewise_node_locations(n_basis: int):
    dx = 1.0/n_basis
    return onp.linspace(0, 1-dx, n_basis) + dx*0.5


def interpolate_piecewise_constant(f: ContinuousParameters, n_basis: int) -> Iterable[JaxTreeType]:
    return [f(t) for t in piecewise_node_locations(n_basis)]


def fem_node_locations(n_basis: int):
    if n_basis == 1:
        return onp.array([0.5])
    else:
        return onp.linspace(0, 1, n_basis)


def interpolate_fem_linear(f: ContinuousParameters, n_basis: int) -> Iterable[JaxTreeType]:
    return [f(t) for t in fem_node_locations(n_basis)]


INTERPOLATE = {
    'piecewise_constant': interpolate_piecewise_constant,
    'fem_linear': interpolate_fem_linear,
    #'piecewise_linear': piecewise_linear,
    #'poly_linear': poly_linear,    
}


#
# Point cloud projections.
#
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


#
# Function projections.
#
def projection_loss(params_A, params_B, basis_A, basis_B, degree=7, n_cell=10):
    """Loss function that integrates over the depth of the network."""
    Gauss_Z, Gauss_W = onp.polynomial.legendre.leggauss(degree)
    phi_A = basis_A(params_A)
    phi_B = basis_B(params_B)
    cell_xs = onp.linspace(0, 1, n_cell)
    integration_cells = [(cell_xs[i], cell_xs[i + 1]) for i in range(n_cell - 1)
                        ]
    losses = []
    for x_1, x_2 in integration_cells:
        for z, w in zip(Gauss_Z, Gauss_W):
            x_z = x_1 + (x_2 - x_1) * (z + 1.0) / 2.0
            integral = 0.5 * (x_2 - x_1) * w * (phi_A(x_z) - phi_B(x_z))**2
            losses.append(integral)
    return jnp.sum(jnp.array(losses))


def _function_project(source_params, source_basis, target_basis, n_basis):
    """Linear function projection is one step of Newton's method."""
    #print('tracing', source_params.shape)
    target_params = jnp.zeros(n_basis)
    n_cell = max(len(source_params), n_basis)
    vG = jax.grad(projection_loss)(target_params,
                                   source_params,
                                   target_basis,
                                   source_basis,
                                   n_cell=n_cell)
    mH = jax.hessian(projection_loss)(target_params,
                                      source_params,
                                      target_basis,
                                      source_basis,
                                      n_cell=n_cell)
    d_params = -jnp.linalg.solve(mH, vG)
    return d_params


function_project = jax.jit(_function_project, static_argnums=[1, 2, 3])

# f_ = lambda x_: function_project(x_, source_basis, target_basis, n_basis)
_function_project_mapped = jax.vmap(function_project, in_axes=(-1, None, None, None), out_axes=-1)
function_project_mapped = jax.jit(_function_project_mapped, static_argnums=[1, 2, 3])


def function_project_array(source_params, source_basis, target_basis, n_basis):
    ys_stack = jnp.array(source_params)
    ys_flat = ys_stack.reshape(ys_stack.shape[0], -1)
    nodes = function_project_mapped(ys_flat, source_basis, target_basis, n_basis)
    return list(nodes.reshape((n_basis,) + ys_stack.shape[1:]))


def function_project_tree(source_params, source_basis, target_basis, n_basis):

    def function_project_list(*args):
        return list(
            function_project_array(args, source_basis, target_basis, n_basis))

    out = jax.tree_multimap(function_project_list, *source_params)
    original_struct = jax.tree_structure(source_params[0])
    mapped_struct = jax.tree_structure(list(range(n_basis)))
    return jax.tree_transpose(original_struct, mapped_struct, out)
