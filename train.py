"""
Train baseline models to demonstrate JumpReLU.
"""


from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
import os

from models import *

import timeit

#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='MNIST Example')
#
parser.add_argument('--name', type=str, default='mnist', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='learning rate (default: 0.01)')
#
parser.add_argument('--lr-decay', type=float, default=0.2, help='learning rate ratio')
#
parser.add_argument('--lr-schedule', type=str, default='normal', help='learning rate schedule')
#
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[30,60,80], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
#
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
#
parser.add_argument('--arch', type=str, default='LeNetLike',  help='choose the architecture')
#
parser.add_argument('--depth', type=int, default=34, help='choose the depth for wide resnet')
#

parser.add_argument('--resume', type=int, default=0, help='resume pre trained model')

parser.add_argument('--resume_path', type=str, default='mnist_result/JumpNetbaseline1.pkl', help='choose an existing model')

parser.add_argument('--widen_factor', type=int, default=4, metavar='E', help='Widen factor')

parser.add_argument('--dropout', type=float, default=0.0, metavar='E', help='Dropout rate')

#
args = parser.parse_args()



#==============================================================================
# set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.name + '_result'):
    os.mkdir(args.name + '_result')

for arg in vars(args):
    print(arg, getattr(args, arg))

#==============================================================================
# get dataset
#==============================================================================
train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)
print('data is loaded')


#==============================================================================
# get model and optimizer
#==============================================================================
model_list = {
        'LeNetLike': LeNetLike(),
        'AlexLike': AlexLike(),
        'ResNet': ResNet(depth=20),
        'WideResNet': WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropout_rate=args.dropout, num_classes=10, level=1), 
}


model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)


if args.resume == 1:
    model.load_state_dict(torch.load(args.resume_path))
    model.train()

#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')
print(model)

#==============================================================================
# Run
#==============================================================================
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

inner_loop = 0
num_updates = 0

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        model.train()
        data, target = data.cuda(), target.cuda()        
        output = model(data)        
        
        loss = criterion(output, target) 
        loss.backward()
        train_loss += loss.item()*target.size()[0]
        total_num += target.size()[0]
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        
        optimizer.step()
        optimizer.zero_grad()
            

    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('Validation Error: ', correct / total_num) 
    
    # schedule learning rate decay    
    optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)



torch.save(model.state_dict(), args.name + '_result/'+args.arch+'_baseline'+'.pkl')  
 
