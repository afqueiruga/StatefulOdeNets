import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=True),)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.conv2(self.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, level=1):
        super().__init__()
        self.in_planes = 16

        self.relu = nn.ReLU(inplace=True)

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic,
                                       nStages[1],
                                       n,
                                       dropout_rate,
                                       stride=1)
        self.layer2 = self._wide_layer(wide_basic,
                                       nStages[2],
                                       n,
                                       dropout_rate,
                                       stride=2)
        self.layer3 = self._wide_layer(wide_basic,
                                       nStages[3],
                                       n,
                                       dropout_rate,
                                       stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.mode = 'normal'

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        out = self.conv1(x)
        for i in range(len(self.layer1)):
            out = self.layer1[i](out)

        for i in range(len(self.layer2)):
            out = self.layer2[i](out)

        for i in range(len(self.layer3)):
            out = self.layer3[i](out)

        out = self.relu(self.bn1(out))

        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out
