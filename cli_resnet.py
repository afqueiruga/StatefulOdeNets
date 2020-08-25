"""
Train baselines
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
import timeit

import os

from residual_net.utils import *
from residual_net import ResNetv2, WideResNet





#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='CIFAR-10 Example')
#
parser.add_argument('--name', type=str, default='cifar10', metavar='N', help='dataset')
#
parser.add_argument('--nclass', type=int, default=10, metavar='S', help='number of classes')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=180, metavar='N', help='number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
#
parser.add_argument('--lr-decay', type=float, default=0.1, help='learning rate ratio')
#
parser.add_argument('--lr-schedule', type=str, default='normal', help='learning rate schedule')
#
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80, 120, 160], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
#
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
#
parser.add_argument('--arch', type=str, default='ResNet',  help='choose the architecture')
#
parser.add_argument('--depth_res', type=int, default=20, help='choose the depth for wide resnet')
#
parser.add_argument('--depth_wide', type=int, default=34, help='choose the depth for wide resnet')
#
parser.add_argument('--widen_factor', type=int, default=4, metavar='E', help='Widen factor')
#
parser.add_argument('--dropout', type=float, default=0.0, metavar='E', help='Dropout rate')
#
parser.add_argument('--use_batch_norms', type=bool, default=True, metavar='N', help='include batch norm layers')
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
        'ResNet': ResNetv2(depth=args.depth_res, num_classes=args.nclass),
        'WideResNet': WideResNet(depth=args.depth_wide, widen_factor=args.widen_factor, dropout_rate=args.dropout, num_classes=args.nclass, level=1),
}


model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)



#==============================================================================
# Model summary
#==============================================================================
print(model)
print('**** Setup ****')
n_params = sum(p.numel() for p in model.parameters())
print('Total params: %.2fk ; %.2fM' % (n_params*10**-3, n_params*10**-6))
print('************')

#==============================================================================
# Run
#==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

inner_loop = 0
num_updates = 0

times = []

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    t0 = timeit.default_timer()

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

    times.append(timeit.default_timer()  - t0)
    model.eval()
    correct = 0
    total_num = 0
    n_print = 5
    if epoch == 0 or (epoch+1) % n_print == 0:
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
        print('Validation Error: ', correct / total_num)

    # schedule learning rate decay
    optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)


print('Average time (without eval): ', np.mean(times))
print('Total time (without eval): ', np.sum(times))

if args.arch == 'ResNet':
	torch.save(model, args.name + '_result/' +args.arch + '_' + str(args.depth_res) + '.pkl')
else:
	torch.save(model, args.name + '_result/' +args.arch + '_' + str(args.depth_wide) + '.pkl')
