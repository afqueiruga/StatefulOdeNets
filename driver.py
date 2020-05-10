"""This is the top level training program. It can be scripted in python or run from the CLI through cli.py."""
import os
from matplotlib import pylab as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from odenet import datasets
from odenet.helper import set_seed, get_device, which_device
from odenet import odenet_cifar10
from odenet import refine_train


def init_params(net):
    '''Init layer parameters using a Kaiming scheme.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0.0)

def do_a_train_set(
    dataset, which_model, ALPHA, scheme, use_batch_norms,
    initial_time_d, time_epsilon, n_time_steps_per,
    N_epochs, N_adapt, lr,
    lr_decay=0.1, epoch_update=None, weight_decay=1e-5,
    use_adjoint=False,
    use_kaiming=False,
    use_skip_init=False,
    refine_variance=0.0,
    seed=1, device=None):
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
    try:
        os.mkdir('results')
        print("Making directory ", "results.")
    except:
        print("Directory ", "results", " already exists.")
    fname =  f'results/odenet_{dataset}_{which_model}_ARCH_{ALPHA}_{use_batch_norms}_{"SkipInit" if use_skip_init else "NoSkip"}_{scheme}_{initial_time_d}_{time_epsilon}_{n_time_steps_per}_LEARN_{lr}_{N_epochs}_{N_adapt}_{refine_variance}_{"Adjoint" if use_adjoint else "Backprop"}_{"KaimingInit" if use_kaiming else "NormalInit"}_SEED_{seed}.pkl'
    print("Working on ", fname)
    set_seed(seed)
    device = get_device(device)

    refset,trainset,trainloader,testset,testloader = datasets.get_dataset(dataset,root='../data/')

    if dataset=="CIFAR10":
        in_channels=3
    elif dataset=="FMNIST":
        in_channels=1
    else:
        in_channels=3
        
    if which_model == "ThreePerSegment":
        model = odenet_cifar10.ODEResNet(
            ALPHA=ALPHA,
            scheme=scheme,
            time_d=initial_time_d,
            in_channels=in_channels,
            use_batch_norms=use_batch_norms,
            time_epsilon=time_epsilon,
            n_time_steps_per=n_time_steps_per,
            use_adjoint=use_adjoint
        ).to(device)
    elif which_model == "SingleSegment":
        model = odenet_cifar10.ODEResNet_SingleSegment(
            ALPHA=ALPHA,
            scheme=scheme,
            time_d=initial_time_d,
            in_channels=in_channels,
            use_batch_norms=use_batch_norms,
            time_epsilon=time_epsilon,
            n_time_steps_per=n_time_steps_per,
            use_skip_init=use_skip_init,
            use_adjoint=use_adjoint
        ).to(device)
    else:
        raise RuntimeError("Unknown model name specified")
    if use_kaiming:
        model.apply(init_params)
    #model = torch.nn.DataParallel(model)
    
    print(model)
    print('**** Setup ****')
    n_params = sum(p.numel() for p in model.parameters())
    print('Total params: %.2fk ; %.2fM' % (n_params*10**-3, n_params*10**-6))

    print('************')
    
    res = refine_train.train_adapt(
        model, trainloader, testloader, torch.nn.CrossEntropyLoss(),
        N_epochs, N_adapt, lr=lr, lr_decay=lr_decay, epoch_update=epoch_update, weight_decay=weight_decay,
        refine_variance=refine_variance,
        device=device)
    
    torch.save(res, fname)
    print("Wrote", fname)
    #plt.semilogy(res[1])
    #for r in res[2]:
    #    plt.axvline(r,color='k')
    return res
