from torch import nn
from .odenet import *

class ODEResNet(nn.Module):
    """Better idea."""
    def __init__(self, method='euler', time_d=1):
        super(ODEResNet,self).__init__()
        self.method = method
        self.time_d = time_d
        self.net = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            ODEBlock(ShallowConv2DODE(1,16,16),method=method),
            #nn.MaxPool2d(2), # Downsampale
            nn.Conv2d(16,32,kernel_size=1,stride=2,bias=False),
            ODEBlock(ShallowConv2DODE(1,32,32),method=method),
            nn.Conv2d(32,64,kernel_size=1,stride=2,bias=False),
            ODEBlock(ShallowConv2DODE(1,64,64),method=method),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64,10),
        )
    def forward(self,x):
        return self.net(x)
    def refine(self):
        new = copy.deepcopy(self)
        for i in range(len(self.net)):
            new.net[i] = odenet.refine(self.net[i])
        return new
