"""These functions are designed for mutatable functions.

A Flax model functionally emits both the mutated variables:

F(p, x, mutable=False) = dx/dt
F(p, x, mutable=True) = p', dx/dt

These schemes are aware of that statefulness:

"""

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
    t1 = t0
    k1, state1 = f(t1, x)
    return x + Dt * k1, (t1,), (state1,)


def Midpoint(f: RateEquation,
             x: ArrayType,
             t0: float = 0,
             Dt: float = 1.0) -> ArrayType:
    """Explicit Midpoint, a two stage Runge Kutta, O(Dt^2)."""
    t1 = t0
    k1, state1 = f(t1, x)
    x1 = x + 0.5 * Dt * k1
    t2 = t0 + 0.5 * Dt  # t = 1/2
    k2, state2 = f(t2, x1)
    return x + Dt * k2, (t1, t2), (state1, state2)


def RK4(f: RateEquation,
        x: ArrayType,
        t0: float = 0,
        Dt: float = 1.0) -> ArrayType:
    """The 'classic' RK4, a four stage Runge Kutta, O(Dt^4)."""
    t1 = t0  # t = 0+ (inside of domain)
    k1, state1 = f(t1, x)
    x1 = x + 0.5 * Dt * k1
    t2 = t0 + 0.5 * Dt  # t = 1/2
    k2, state2 = f(t2, x1)
    x2 = x + 0.5 * Dt * k2
    t3 = t0 + 0.5 * Dt  # t = 1/2
    k3, state3 = f(t3, x2)
    x3 = x + Dt * k3
    t4 = t0 + Dt - DT_OPEN_SET_EPS  # t = 1- (inside domain)
    k4, state4 = f(t4, x3)
    x_out = x + Dt * (1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 +
                      1.0 / 6.0 * k4)
    return x_out, (t1, t2, t3, t4), (state1, state2, state3, state4)


def RK4_38(f: RateEquation,
           x: ArrayType,
           t0: float = 0,
           Dt: float = 1.0) -> ArrayType:
    """The 3/8s RK4, a four stage Runge Kutta, O(Dt^4)."""
    t1 = t0  # t = 0+ (inside of domain)
    k1, state1 = f(t1, x)  # t = 0+ (inside of domain)
    x1 = x + 1.0 / 3.0 * Dt * k1
    t2 = t0 + 1.0 / 3.0 * Dt
    k2, state2 = f(t2, x1)  # t = 1/3
    x2 = x + Dt * (-1.0 / 3.0 * k1 + 1.0 * k2)
    t3 = t0 + 2.0 / 3.0 * Dt
    k3, state3 = f(t3, x2)  # t = 2/2
    x3 = x + Dt * (k1 - k2 + k3)
    t4 = t0 + Dt - DT_OPEN_SET_EPS
    k4, state4 = f(t4, x3)  # t = 1-
    x_out = x + Dt * (1.0 / 8.0 * k1 + 3.0 / 8.0 * k2 + 3.0 / 8.0 * k3 +
                      1.0 / 8.0 * k4)
    return x_out, (t1, t2, t3, t4), (state1, state2, state3, state4)


SCHEME_TABLE = {
    'Euler': Euler,
    'Midpoint': Midpoint,
    'RK4': RK4,
    'RK4_38': RK4_38,
}


def StateOdeIntegrateFast(f: RateEquation,
                          x: ArrayType,
                          scheme: Union[str, IntegrationScheme] = Euler,
                          n_step: int = 10) -> ArrayType:
    try:
        scheme = SCHEME_TABLE[scheme]
    except:
        pass
    dt = 1.0 / n_step
    all_ts, states = [], []
    for t in onp.arange(0, 1, dt):
        x, ts, state = scheme(f, x, t, dt)
        all_ts.extend(ts)
        states.extend(state)
    return x, all_ts, states


def StateOdeIntegrateWithPoints(f: RateEquation,
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
        state, x = scheme(f, x, t, dt)
        xs.append(onp.array(x))
    return xs
