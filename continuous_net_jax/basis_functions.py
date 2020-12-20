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
