from .continuous_types import *

from matplotlib import pylab as plt
import numpy as onp


def plot_fun(f: Callable[[float], float]):
    """Plot a function from [0,1]."""
    ts = onp.linspace(0,1,200)
    y = onp.array([f(t) for t in ts])
    plt.plot(ts,y)


def plot_piecewise_nodes(nodes: Iterable[float]):
    n = len(nodes)
    dt = 1 / n
    ts = onp.linspace(0.5*dt, 1.0-0.5*dt, n)
    plt.plot(ts, nodes, 'o')


def plot_fem_nodes(nodes: Iterable[float]):
    plt.plot(onp.linspace(0, 1, len(nodes)), nodes, 'o')
