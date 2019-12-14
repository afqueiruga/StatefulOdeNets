from torch import nn
import math
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
#            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),


            nn.Conv2d(ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*ALPHA),
            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),


            nn.Conv2d(2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA), 
            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),

            nn.Conv2d(4*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA), 
            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),


            nn.AdaptiveAvgPool2d(1),
            #nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(4*ALPHA,10),
        )
            
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #elif isinstance(m, nn.BatchNorm2d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()            
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)              
                        
            
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

