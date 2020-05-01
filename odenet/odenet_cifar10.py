from torch import nn
import math
from .ode_models import *


class ODEResNet(nn.Module):
    """Multiple ODEBlocks per segment."""
    def __init__(self,
                 ALPHA=16,
                 scheme='euler',
                 time_d=1,
                 in_channels=3,
                 time_epsilon=1.0,
                 use_batch_norms=True,
                 use_adjoint=False):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        if time_d%3 != 0:
            print("Uh-oh: This class wanted time_d divisible by three!")
        # This macro lets us make 3 of them concisely without typos
        _macro = lambda _alpha : \
            ODEBlock(
                ShallowConv2DODE(
                    time_d//3,
                    _alpha,
                    _alpha,
                    epsilon=time_epsilon,
                    use_batch_norms=use_batch_norms),
                n_time_steps=time_d*n_time_steps_per,
                method=method,
                use_adjoint=use_adjoint)
        # The full resnet, with three segments with 3 macros each
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, ALPHA, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ALPHA),
            nn.ReLU(),
            # Segment 1
            _macro(ALPHA),
            _macro(ALPHA),
            _macro(ALPHA),

            nn.Conv2d(
                ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*ALPHA),
            # Segment 2
            _macro(2*ALPHA),
            _macro(2*ALPHA),
            _macro(2*ALPHA),
            
            nn.Conv2d(
                2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA),
            # Segment 3
            _macro(4*ALPHA),
            _macro(4*ALPHA),
            _macro(4*ALPHA),

#            nn.Conv2d(4*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
#            nn.BatchNorm2d(4*ALPHA), 
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d*2, method=method, use_adjoint=use_adjoint),
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
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()     
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


class ODEResNet_SingleSegment(nn.Module):
    """Uses one OdeBlock per segment."""
    def __init__(self,
                 ALPHA=16,
                 scheme='euler',
                 time_d=1,
                 in_channels=3,
                 use_batch_norms=True,
                 time_epsilon=1.0,
                 n_time_steps_per=1,
                 use_adjoint=False):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        # This macro lets us make 3 of them concisely without typos
        _macro = lambda _alpha : \
            ODEBlock(
                ShallowConv2DODE(
                    time_d,
                    _alpha,
                    _alpha,
                    epsilon=time_epsilon,
                    use_batch_norms=use_batch_norms),
                n_time_steps=time_d*n_time_steps_per,
                scheme=scheme,
                use_adjoint=use_adjoint)
        # The full resnet, with three segments of the above macro
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, ALPHA, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(ALPHA),
            nn.ReLU(),
            _macro(ALPHA),
            nn.Conv2d(
                ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*ALPHA), 
            #nn.ReLU(),
            _macro(2*ALPHA),
            
            nn.Conv2d(
                2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA),
            #nn.ReLU(),
            _macro(4*ALPHA),
            
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


class ODEResNet2(nn.Module):
    """Not sure what was going on here."""
    def __init__(self, ALPHA=16, scheme='euler', time_d=1,
                in_channels=3, use_adjoint=False, use_batch_norms=True):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        self.net = nn.Sequential(
            
            nn.Conv2d(in_channels, ALPHA, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ALPHA),
            nn.ReLU(),
            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
                     N_time=time_d, scheme=scheme, use_adjoint=use_adjoint),
            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
                     N_time=time_d, scheme=scheme, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, ALPHA, ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),


#            nn.Conv2d(ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
#            nn.BatchNorm2d(2*ALPHA),
#            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
#                     N_time=2, method=method, use_adjoint=use_adjoint),
##            ODEBlock(ShallowConv2DODE(time_d, 2*ALPHA, 2*ALPHA),
##                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#
#
#            nn.Conv2d(2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
#            nn.BatchNorm2d(4*ALPHA), 
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=2, method=method, use_adjoint=use_adjoint),
##            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
##                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#
#            nn.Conv2d(4*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
#            nn.BatchNorm2d(4*ALPHA), 
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=time_d, method=method, use_adjoint=use_adjoint),
#            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
#                     N_time=2, method=method, use_adjoint=use_adjoint),
##            ODEBlock(ShallowConv2DODE(time_d, 4*ALPHA, 4*ALPHA),
##                     N_time=time_d, method=method, use_adjoint=use_adjoint),

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