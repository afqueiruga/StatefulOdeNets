"""This is the top level training program. It can be scripted in python or run from the CLI through cli.py."""
import os
from matplotlib import pylab as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from continuous_net import datasets
from continuous_net.helper import set_seed, get_device, which_device
from continuous_net import refine_train
from continuous_net import continuous_net, wide_continuous_net

SAVE_DIR = 'results'
#******************************************************************************
# Create folder to save results
#******************************************************************************
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def do_a_train_set(
    dataset,
    which_model,
    ALPHA,
    widen_factor,
    scheme,
    use_batch_norms,
    initial_time_d,
    time_epsilon,
    n_time_steps_per,
    N_epochs,
    N_adapt,
    lr,
    lr_decay=0.1,
    epoch_update=None,
    weight_decay=5e-4,
    batch_size = 128,
    test_batch_size = 512,
    use_adjoint=False,
    use_skip_init=False,
    refine_variance=0.0,
    seed=1,
    device=None):
    """Set up and train one model, and save it.

    Args:
        dataset: Which dataset to load
        ALPHA: Multiplier for inner width of resnet units
        scheme: One of "euler", "midpoint", or "RK4"
        N_epochs: How many dataset epochs to train over for the whole duration
        N_adapt: A list of epoch numbers at which to refine. Only numbers less than N_epochs are meaningful
        lr: Initial learning rate
        lr_decay: Learning rate decay
        epoch_update: A list of epochs at which to call a learning_rate schedule
        weight_decay: Traditional weight decay parameter
        use_adjoint: Use the adjoint method for backpropogation
        seed: a seed
        device: which device to use
    """

    fname = SAVE_DIR+f'/continuousnet-{dataset}-{which_model}-{scheme}-{initial_time_d}-{n_time_steps_per}-{N_epochs}-{N_adapt}-{refine_variance}-{"Adjoint" if use_adjoint else "Backprop"}-SEED-{seed}.pkl'

    print("Working on ", fname)
    set_seed(seed)
    device = get_device(device)

    if time_epsilon is None:
        time_epsilon = initial_time_d

    refset,trainset,trainloader,testset,testloader = \
        datasets.get_dataset(dataset,root='../data/', batch_size=batch_size, test_batch_size=test_batch_size)

    if dataset=="CIFAR10":
        out_classes = 10
        in_channels=3
    elif dataset=="CIFAR100":
        out_classes = 100
        in_channels=3
    elif dataset=="tinyimagenet":
        out_classes = 200
        in_channels=3
    elif dataset=="FMNIST":
        out_classes=10
        in_channels=1

    if which_model == "ContinuousNet":
        model = continuous_net.ContinuousNet(
            ALPHA=ALPHA,
            scheme=scheme,
            time_d=initial_time_d,
            in_channels=in_channels,
            out_classes=out_classes,
            use_batch_norms=use_batch_norms,
            time_epsilon=time_epsilon,
            n_time_steps_per=n_time_steps_per,
            use_skip_init=use_skip_init,
            use_adjoint=use_adjoint,
            activation_before_conv=False,
        ).to(device)
    elif which_model == "ContinuousNetActFirst":
        model = continuous_net.ContinuousNet(
            ALPHA=ALPHA,
            scheme=scheme,
            time_d=initial_time_d,
            in_channels=in_channels,
            out_classes=out_classes,
            use_batch_norms=use_batch_norms,
            time_epsilon=time_epsilon,
            n_time_steps_per=n_time_steps_per,
            use_skip_init=use_skip_init,
            use_adjoint=use_adjoint,
            activation_before_conv=True,
        ).to(device)
    elif which_model == "WideContinuousNet":
        model = wide_continuous_net.WideContinuousNet(
            ALPHA=ALPHA,
            widen_factor=widen_factor,
            scheme=scheme,
            time_d=initial_time_d,
            in_channels=in_channels,
            out_classes=out_classes,
            use_batch_norms=use_batch_norms,
            time_epsilon=time_epsilon,
            n_time_steps_per=n_time_steps_per,
            use_skip_init=use_skip_init,
            use_adjoint=use_adjoint,
            activation_before_conv=True,
        ).to(device)

    else:
        raise RuntimeError("Unknown model name specified")


    print(model)
    print('**** Setup ****')
    n_params = sum(p.numel() for p in model.parameters())
    print('Total params: %.2fk ; %.2fM' % (n_params*10**-3, n_params*10**-6))
    print('************')

    res = refine_train.train_adapt(
        model, trainloader, testloader, torch.nn.CrossEntropyLoss(),
        N_epochs, N_adapt, lr=lr, lr_decay=lr_decay, epoch_update=epoch_update, weight_decay=weight_decay,
        refine_variance=refine_variance,
        device=device,
        SAVE_DIR=SAVE_DIR, fname=fname)

    try:
        os.mkdir(SAVE_DIR)
        print("Making directory ", "results.")
    except:
        print("Directory ", SAVE_DIR, " already exists.")
    torch.save(res, fname)
    print("Wrote", fname)
    return res
