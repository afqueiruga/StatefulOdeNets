"""
This file performs the manifestation invariance tests on RefineNets to 
reproduce Figure 4.
"""
from collections import defaultdict
import functools
from glob import glob
import json
import os
import pickle
import re
from typing import List

from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import torch
try:
    import tqdm
    progress = tqdm.tqdm
except ModuleNotFoundError:
    progress = lambda x, *args : x

from refine_net import refine_train
from refine_net import ode_models
from refine_net import datasets
from refine_net import plotting
from refine_net import helper

# The dataset used for testing
refset,trainset,trainloader,testset,testloader = \
    datasets.get_dataset("CIFAR10",root='../data/')
# How many time steps to use
N_time_max = 4
# Which files to load
dirs = glob('results/refinenet-CIFAR10*')
dirs = sorted(dirs, key = lambda x : os.stat(x).st_mtime)[:]
# Load the torch pickles, and load them onto one device
DEV = "cuda:3"
results = { name: torch.load(name, map_location={f"cuda:{i}":DEV for i in range(4)}) for name in dirs }


def legend_name(fname):
    """Strip the prefix from filenames to print the architecture"""
    return re.search(r"ARCH(.*)", fname)[1]

def set_ode_config(model, n_steps, scheme, use_adjoint=False):
    """Change the configuration of OdeBlocks in the model."""
    for net_idx in range(len(model.net)):
        try:
            model.net[net_idx].set_n_time_steps(n_steps)
            model.net[net_idx].scheme = scheme
        except AttributeError:
            pass

@torch.no_grad()
def calc_acc_for_scheme_nt(name, scheme, nt):
    """Evaluate the Test Accuracy for a given manifestation."""
    mod = results[name].model_list[-1]
    mod.eval()
    set_ode_config(mod, nt, scheme)
    return [refine_train.calculate_accuracy(mod, testloader)]

def calculate_plot(nt_max):
    """
    Loop over many nts and schemes to compute test errors as a function of graph
    manifestation.
    
    (This function is almost as expensive as training a model; the results are
    saved to a pickle below for plot formatting. the version released 
    publically will use a SQL cacheing library written by the authors.)"""
    test_sweep = defaultdict( lambda : {} )
    with torch.no_grad():
        for name in results:
            result = results[name]
            accs_for_scheme = {}
            for scheme in ('euler','rk4','rk4_classic',):
                accs = []
                nts = range(1,nt_max,1)
                for nt in progress( nts, f"{legend_name(name)}, {scheme}"):
                    accs.append((nt, calc_acc_for_scheme_nt(name, scheme, nt)[0]))
                accs_for_scheme[scheme] = accs
            test_sweep[name].update(accs_for_scheme)
    return test_sweep

# Do it
test_sweep = calculate_plot(N_time_max)
# Save a pickle to plot it later.
pickle.dump( dict(test_sweep), open( "manifestation_plot_points.pkl", "wb" ) )

# Plot the results, using the hard-storage.
plot_points = pickle.load(  open( "manifestation_plot_points.pkl", "rb" )) 
plt.figure(figsize=(12,8))
for name,vals in plot_points.items():
    for scheme,accs in vals.items():
        if "euler"==scheme:
            original_scheme = "euler"
        else:
            original_scheme = "rk4_classic"
        if ("euler" in name and "euler"==scheme) or ("rk4_classic" in name and "rk4_classic"==scheme):
            plt.subplot(1,2,1)
            plt.ylim(0,1)
            plt.ylabel("Test Accuracy")
            plt.xlabel("Computational Graph Depth $N_t=1/\Delta t$")
        else:
            plt.subplot(1,2,2)
            plt.ylim(0,1)
            plt.xlabel("Computational Graph Depth $N_t=1/\Delta t$")
        plt.plot(*zip(*accs),label=f"{original_scheme}->{scheme}")
plt.legend()
plt.savefig("manifestation_invariance.pdf")