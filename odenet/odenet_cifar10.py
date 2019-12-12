from torch import nn
from .odenet import *

class ODEResNet(nn.Module):
    """Better idea."""
    def __init__(self, ALPHA=16, method='euler', time_d=1,
                in_channels=3, use_adjoint=False):
        super(ODEResNet,self).__init__()
        self.method = method
        self.time_d = time_d
        self.net = nn.Sequential(
            
            nn.Conv2d(in_channels, ALPHA, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ALPHA),
            nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
            #ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
            #         N_time=time_d, method=method, use_adjoint=use_adjoint),
            #ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
            #         N_time=time_d, method=method, use_adjoint=use_adjoint),
            nn.Conv2d(ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*ALPHA),
            nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
            #ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
            #         N_time=time_d, method=method, use_adjoint=use_adjoint),
            #ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
            #         N_time=time_d, method=method, use_adjoint=use_adjoint),
            nn.Conv2d(2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA), 
            nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
            #ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
            #         N_time=time_d, method=method, use_adjoint=use_adjoint),
            #ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
            #         N_time=time_d, method=method, use_adjoint=use_adjoint),
            nn.AdaptiveAvgPool2d(1),
            #nn.AvgPool2d(8),
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

class ODEResNet_new(nn.Module):
    """Better idea."""
    def __init__(self, ALPHA=16, method='euler', time_d=1,
                in_channels=3, use_adjoint=False):
        super(ODEResNet,self).__init__()
        self.method = method
        self.time_d = time_d
        self.net = nn.Sequential(
            
            nn.Conv2d(in_channels, ALPHA, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(ALPHA),
            nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
            
            nn.Conv2d(ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*ALPHA), 
            #nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
            
            nn.Conv2d(2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA),
            #nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
            
            nn.AdaptiveAvgPool2d(1),
            #nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(4*ALPHA,10),
        )
    def forward(self,x):
        return self.net(x)
    def refine(self):
        new = ODEResNet.__new__(ODEResNet)
        new.time_d = 2*self.time_d
        new.method = self.method
        new.net = nn.Sequential(*[ refine(mod) for mod in self.net])
        return new

