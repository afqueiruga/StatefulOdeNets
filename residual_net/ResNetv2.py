# Resnet v2 
# Note that this is a ResNetv2 implementation that we adapted based on code provided by https://github.com/bearpaw/pytorch-classification
#

from __future__ import absolute_import

import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)

def NoSequential(*args):
    """Filters Nones as no-ops when making ann.Sequential to allow for architecture toggling."""
    net = [ arg for arg in args if arg is not None ]
    return nn.Sequential(*net)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None, with_bn = True):
        super(BasicBlock, self).__init__()
        

        self.with_bn = with_bn
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.with_bn: self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = False)
        self.conv2 = conv3x3(planes, planes)
        if self.with_bn: self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)        
        if self.with_bn: out = self.bn1(out)

        out = self.conv2(out)  
        out = self.relu(out)        
        if self.with_bn: out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, with_bn = True):
        super(Bottleneck, self).__init__()

        self.with_bn = with_bn
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        if self.with_bn: self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        if self.with_bn: self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size = 1, bias = False)
        if self.with_bn: self.bn3 = nn.BatchNorm2d(planes * 4)
       	self.relu = nn.ReLU(inplace = False)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.with_bn: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_bn: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.with_bn: out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        

        return out

ALPHA_ = 1
class ResNet(nn.Module):

    def __init__(self, depth, num_classes = 10, use_batch_norms=True):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 500 else BasicBlock

        self.with_bn = use_batch_norms
        self.inplanes = 16 * ALPHA_
        self.conv1 = nn.Conv2d(3, 16 * ALPHA_, kernel_size = 3, padding = 1, bias = False)
        
        if self.with_bn: self.bn1 = nn.BatchNorm2d(16 * ALPHA_)
        
        self.relu = nn.ReLU(inplace = True)
        self.layer1 = self._make_layer(block, 16 * ALPHA_, n, with_bn=self.with_bn)
        self.layer2 = self._make_layer(block, 32 * ALPHA_, n, stride = 2, with_bn=self.with_bn)
        self.layer3 = self._make_layer(block, 64 * ALPHA_, n, stride = 2, with_bn=self.with_bn)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * ALPHA_* block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.mode = 'normal'

    def _make_layer(self, block, planes, blocks, stride = 1, with_bn=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = NoSequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                          nn.BatchNorm2d(planes * block.expansion) if with_bn else None,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, with_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_bn=with_bn))

        return nn.Sequential(*layers)


    def forward(self, x):
        bs = x.size(0)
        
        x = self.conv1(x)
        if self.with_bn: x = self.bn1(x)
        x = self.relu(x)    # 32x32

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
