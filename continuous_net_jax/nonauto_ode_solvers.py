from typing import Any, Callable, Dict, List, Iterable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp

import flax

# JAX doesn't have a good type system yet, so this is for readability.
ArrayType = Any
JaxTreeType = Union[ArrayType, Iterable['JaxTreeType'], Dict[Union[str, int],
                                                             'JaxTreeType']]

# Just the type signature of a normal jax function, or a flax.nn.Module.call.
RateEquation = Callable[[JaxTreeType, ArrayType], ArrayType]
# A general depth basis function.
BasisFunction = Callable[[Iterable[JaxTreeType], float, int], JaxTreeType]
# An instance of a depth function.
ContinuousParameters = Callable[[float], JaxTreeType]
# Integration scheme function.
IntegrationScheme = Callable[[ContinuousParameters, float, RateEquation, float],
                             JaxTreeType]


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


# Integrators
# Rigorously, one step only evaluates f on the open set [t0, t0+Dt). We substract
# this eps to keep evaluations of params_of_t inside of that interval to align
# with domains.
DT_OPEN_SET_EPS = 1.0e-5


def Euler(params_of_t: ContinuousParameters,
          x: ArrayType,
          t0: float = 0,
          f: RateEquation = None,
          Dt: float = 1.0) -> ArrayType:
    """Forward Euler, O(Dt)."""
    return x + Dt * f(params_of_t(t0), x)


def Midpoint(params_of_t: ContinuousParameters,
             x: ArrayType,
             t0: float = 0,
             f=None,
             Dt: float = 1.0) -> ArrayType:
    """Explicit Midpoint, a two stage Runge Kutta, O(Dt^2)."""
    k1 = f(params_of_t(t0), x)
    x1 = x + 0.5 * Dt * k1  # t = 1/2
    return x + Dt * f(params_of_t(t0 + 0.5 * Dt), x1)


def RK4(params_of_t: ContinuousParameters,
        x: ArrayType,
        t0: float = 0,
        f=None,
        Dt: float = 1.0):
    """The 'classic' RK4, a four stage Runge Kutta, O(Dt^4)."""
    k1 = f(params_of_t(t0), x)  # t = 0+ (inside of domain)
    x1 = x + 0.5 * Dt * k1
    k2 = f(params_of_t(t0 + 0.5 * Dt), x1)  # t = 1/2
    x2 = x + 0.5 * Dt * k2
    k3 = f(params_of_t(t0 + 0.5 * Dt), x2)  # t = 1/2
    x3 = x + Dt * k3
    k4 = f(params_of_t(t0 + Dt - DT_OPEN_SET_EPS),
           x3)  # t = 1- (inside of domain)
    return x + Dt * (1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 +
                     1.0 / 6.0 * k4)


def OdeIntegrateFast(params_of_t: ContinuousParameters,
                     x: ArrayType,
                     f: RateEquation,
                     scheme: IntegrationScheme = Euler,
                     n_step: int = 10) -> ArrayType:
    dt = 1.0 / n_step
    for t in onp.linspace(0, 1, n_step):
        x = scheme(params_of_t, x, t, f, dt)
    return x


def OdeIntegrateWithPoints(params_of_t: ContinuousParameters,
                           x: ArrayType,
                           f: RateEquation,
                           scheme: IntegrationScheme = Euler,
                           n_step: int = 10) -> List[ArrayType]:
    dt = 1.0 / n_step
    xs = [onp.array(x)]
    for t in onp.linspace(0, 1, n_step):
        x = scheme(params_of_t, x, t, f, dt)
        xs.append(onp.array(x))
    return xs
