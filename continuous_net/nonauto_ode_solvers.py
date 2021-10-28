import jax
import jax.numpy as jnp
import numpy as onp

import flax

from .continuous_types import *

# Integrators
# Rigorously, one step only evaluates f on the open set [t0, t0+Dt). We substract
# this eps to keep evaluations of params_of_t inside of that interval to align
# with domains.
DT_OPEN_SET_EPS = 1.0e-5


def Euler(f: RateEquation,
          x: ArrayType,
          t0: float = 0,
          Dt: float = 1.0) -> ArrayType:
    """Forward Euler, O(Dt)."""
    return x + Dt * f(t0, x)


def Midpoint(f: RateEquation,
             x: ArrayType,
             t0: float = 0,
             Dt: float = 1.0) -> ArrayType:
    """Explicit Midpoint, a two stage Runge Kutta, O(Dt^2)."""
    k1 = f(t0, x)
    x1 = x + 0.5 * Dt * k1  # t = 1/2
    return x + Dt * f(t0 + 0.5 * Dt, x1)


def RK4(f: RateEquation,
        x: ArrayType,
        t0: float = 0,
        Dt: float = 1.0) -> ArrayType:
    """The 'classic' RK4, a four stage Runge Kutta, O(Dt^4)."""
    k1 = f(t0, x)  # t = 0+ (inside of domain)
    x1 = x + 0.5 * Dt * k1
    k2 = f(t0 + 0.5 * Dt, x1)  # t = 1/2
    x2 = x + 0.5 * Dt * k2
    k3 = f(t0 + 0.5 * Dt, x2)  # t = 1/2
    x3 = x + Dt * k3
    k4 = f(t0 + Dt - DT_OPEN_SET_EPS, x3)  # t = 1- (inside domain)
    return x + Dt * (1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 +
                     1.0 / 6.0 * k4)


def RK4_38(f: RateEquation, x: ArrayType, t0: float = 0, Dt: float = 1.0):
    """The 3/8s RK4, a four stage Runge Kutta, O(Dt^4)."""
    k1 = f(t0, x)  # t = 0+ (inside of domain)
    x1 = x + 1.0 / 3.0 * Dt * k1
    k2 = f(t0 + 1.0 / 3.0 * Dt, x1)  # t = 1/3
    x2 = x + Dt * (-1.0 / 3.0 * k1 + 1.0 * k2)
    k3 = f(t0 + 2.0 / 3.0 * Dt, x2)  # t = 2/2
    x3 = x + Dt * (k1 - k2 + k3)
    k4 = f(t0 + Dt - DT_OPEN_SET_EPS, x3)  # t = 1-
    return x + Dt * (1.0 / 8.0 * k1 + 3.0 / 8.0 * k2 + 3.0 / 8.0 * k3 +
                     1.0 / 8.0 * k4)


SCHEME_TABLE = {
    'Euler': Euler,
    'Midpoint': Midpoint,
    'RK4': RK4,
    'RK4_38': RK4_38,
}


def OdeIntegrateFast(f: RateEquation,
                     x: ArrayType,
                     scheme: Union[str, IntegrationScheme] = Euler,
                     n_step: int = 10) -> ArrayType:
    try:
        scheme = SCHEME_TABLE[scheme]
    except:
        pass
    dt = 1.0 / n_step
    for t in onp.arange(0, 1, dt):
        x = scheme(f, x, t, dt)
    return x


def OdeIntegrateWithPoints(f: RateEquation,
                           x: ArrayType,
                           scheme: Union[str, IntegrationScheme] = Euler,
                           n_step: int = 10) -> List[ArrayType]:
    try:
        scheme = SCHEME_TABLE[scheme]
    except:
        pass
    dt = 1.0 / n_step
    xs = [onp.array(x)]
    for t in onp.arange(0, 1, dt):
        x = scheme(f, x, t, dt)
        xs.append(onp.array(x))
    return xs
