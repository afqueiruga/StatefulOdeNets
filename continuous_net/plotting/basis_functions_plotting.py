from matplotlib import pylab as plt
import numpy as onp

from .basis_functions import *


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


def plot_piecewise_fun(nodes: Iterable[float]):
    plot_fun(piecewise_constant(nodes))
    plot_piecewise_nodes(nodes)


def plot_piecewise_linear_nodes(nodes: Iterable[float]):
    n_cell = len(nodes) // 2
    dt = 1 / n_cell
    ts = []
    for i in range(n_cell):
        ts += [dt*i, dt*(i+1)]
    plt.plot(ts, nodes, 'o')


def plot_piecewise_linear_fun(nodes: Iterable[float]):
    plot_fun(piecewise_linear(nodes))
    plot_piecewise_linear_nodes(nodes)


def plot_fem_nodes(nodes: Iterable[float]):
    plt.plot(onp.linspace(0, 1, len(nodes)), nodes, 'o')


def plot_fem_fun(nodes: Iterable[float]):
    plot_fun(fem_linear(nodes))
    plot_fem_nodes(nodes)


PLOT = {
    'piecewise_constant': plot_piecewise_fun,
    'fem_linear': plot_fem_fun,
    'piecewise_linear': plot_piecewise_linear_fun,
}