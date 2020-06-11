"""
This file performs the manifestation invariance tests on RefineNets to 
reproduce Figure 4.
"""
from collections import defaultdict
import functools
from glob import glob
import json
import os
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

from odenet import refine_train
from odenet import ode_models
from odenet import datasets
from odenet import odenet_cifar10
from odenet import plotting
from odenet import helper

# The dataset used for testing
refset,trainset,trainloader,testset,testloader = \
    datasets.get_dataset("CIFAR10",root='../data/')
# How many time steps to use
N_time_max = 96
# Which files to load
dirs = glob('results_tiny/odenet-CIFAR10*')
dirs = sorted(dirs, key = lambda x : os.stat(x).st_mtime)[:]
# Load the torch pickles, and load them onto one device
DEV = "cuda:0"
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
    
    (This function is almost as expensive as training a model; the version 
    released publically will use a SQL cacheing library written by the authors.)"""
    test_sweep = defaultdict( lambda : {} )
    with torch.no_grad():
        for name in names:
            result = results[name]
            accs_for_scheme = {}
            for scheme in ('euler','rk4','rk4_classic',):
                accs = []s
                nts = range(1,nt_max,1)
                for nt in progress( nts, f"{legend_name(name)}, {scheme}"):
                    accs.append((nt, calc_acc_for_scheme_nt(name, scheme, nt)[0]))
                accs_for_scheme[scheme] = accs
            test_sweep[name].update(accs_for_scheme)
    return test_sweep

# Do it
test_sweep = calculate_plot(N_time_max)


