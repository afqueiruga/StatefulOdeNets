import numpy as np
import torch
import torch.nn as nn

import torch.nn.init as init


from matplotlib import pylab as plt
from odenet import datasets
from odenet.odenet_cifar10 import ODEResNet
from odenet import refine_train

import importlib
importlib.reload(refine_train)
from odenet.helper import set_seed, get_device, which_device
#importlib.reload(odenet_cifar10)

import argparse


#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='odenet', metavar='N', help='Model')
#
parser.add_argument('--dataset', type=str, default='cifar10', metavar='N', help='dataset')
#
parser.add_argument('--lr', type=float, default=1e-1, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=1e-5, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=128, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=128, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='results_det',  help='specify directory to print results to')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 60, 90], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--lr_decay', type=float, default='0.1',  help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--seed', type=int, default='1',  help='Prediction steps')
#
parser.add_argument('--refine', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')
#

args = parser.parse_args()



set_seed(args.seed)
device = get_device()



refset,trainset,trainloader,testset,testloader = datasets.get_dataset('CIFAR10',root='../data/')


def init_params(net):
    '''Init layer parameters.'''
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

def do_a_train_set(ALPHA, method, N_epochs, N_adapt, lr, lr_decay=0.1, epoch_update=[10], weight_decay=1e-5):
    
    model = ODEResNet(ALPHA=ALPHA, method=method, in_channels=3).to(device)
    model.apply(init_params)
    
    #==============================================================================
    # Model summary
    #==============================================================================
    print(model)
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')
    

    res = refine_train.train_adapt(model, trainloader, testloader, torch.nn.CrossEntropyLoss(),
                N_epochs, N_adapt, lr=lr, lr_decay=lr_decay, epoch_update=epoch_update, weight_decay=weight_decay, 
                                   device=device)
    #plt.semilogy(res[1])
    #for r in res[2]:
    #    plt.axvline(r,color='k')
    return res



stash = {}
for method in ['euler','rk4','midpoint']:
#for method in ['euler']:
    res = do_a_train_set(16, method, N_epochs=args.epochs, N_adapt=args.refine, lr=args.lr, lr_decay=args.lr_decay, epoch_update=args.lr_update, weight_decay=args.wd)
    stash[method] = res
    
    
    
torch.save(stash['euler'][0], 'results/resnet_euler.pkl')        
torch.save(stash['rk4'][0], 'results/resnet_rk4.pkl')
