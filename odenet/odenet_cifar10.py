from torch import nn
from .odenet import *

class ODEResNet(nn.Module):
    """Better idea."""
    def __init__(self, ALPHA=16, method='euler', time_d=1,
                in_channels=3):
        super(ODEResNet,self).__init__()
        self.method = method
        self.time_d = time_d
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,ALPHA,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(ALPHA),
            ODEBlock(ShallowConv2DODE(time_d,ALPHA,ALPHA),
                     N_time=time_d,method=method),
            #nn.MaxPool2d(2), # Downsampale
            nn.Conv2d(ALPHA, 2*ALPHA, kernel_size=1,stride=2,bias=False),
            ODEBlock(ShallowConv2DODE(time_d,2*ALPHA,2*ALPHA),
                     N_time=time_d,method=method),
            nn.Conv2d(2*ALPHA, 4*ALPHA,kernel_size=1,stride=2,bias=False),
            ODEBlock(ShallowConv2DODE(time_d,4*ALPHA,4*ALPHA),
                     N_time=time_d,method=method),
            #nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(4*ALPHA,10),
        )
    def forward(self,x):
        return self.net(x)
    def refine(self):
        new = copy.deepcopy(self)
        for i in range(len(self.net)):
            new.net[i] = refine(self.net[i])
        return new
