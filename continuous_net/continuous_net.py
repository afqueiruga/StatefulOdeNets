from torch import nn
import math
from .ode_models import *


def NoSequential(*args):
    """Filters Nones as no-ops when making a nn.Sequential to allow for architecture toggling."""
    net = [ arg for arg in args if arg is not None ]
    return nn.Sequential(*net)


class ContinuousNet(nn.Module):
    """Uses one OdeBlock per segment."""
    def __init__(self,
                 ALPHA=16,
                 scheme='euler',
                 time_d=1,
                 in_channels=3,
                 out_classes=10,
                 use_batch_norms=True,
                 time_epsilon=1.0,
                 n_time_steps_per=1,
                 use_skip_init=False,
                 use_stitch=True,
                 use_adjoint=False,
                 activation_before_conv=True,
                 stitch_epsilon=1.0):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        self.use_batch_norms = use_batch_norms
        
        if activation_before_conv:
            _OdeUnit = ShallowConv2DODE_Flipped
        else:
            _OdeUnit = ShallowConv2DODE
            
        # This macro lets us make 3 of them concisely without typos
        _macro = lambda _alpha : \
            ODEBlock(
                _OdeUnit(
                    time_d,
                    _alpha,
                    _alpha,
                    epsilon=time_epsilon,
                    use_batch_norms=use_batch_norms,
                    use_skip_init=use_skip_init),
                n_time_steps=time_d*n_time_steps_per,
                scheme=scheme,
                use_adjoint=use_adjoint)
        if use_stitch:
            _stitch_macro = lambda _alpha, _beta : \
                ODEStitch(_alpha, _beta, _beta,
                          epsilon=stitch_epsilon,
                          use_batch_norms=use_batch_norms,
                          use_skip_init=use_skip_init)
        else:
            _stitch_macro = lambda _alpha, _beta : \
                nn.Conv2d(_alpha, _beta, kernel_size=1, padding=1, stride=2, bias=False)

        # The full network, with three OdeBlocks (_macro)
        self.net = NoSequential(
            nn.Conv2d(
                in_channels, ALPHA, kernel_size=7, padding=1,bias=False),
            nn.BatchNorm2d(ALPHA) if use_batch_norms else None,
            nn.ReLU(),
            _macro(ALPHA),
            _stitch_macro(ALPHA, 2*ALPHA),
            _macro(2*ALPHA),
            _stitch_macro(2*ALPHA, 4*ALPHA),
            _macro(4*ALPHA),
            nn.BatchNorm2d(4*ALPHA) if activation_before_conv else None,
            nn.ReLU() if activation_before_conv else None,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4*ALPHA,out_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                print('Init Conv2d')
            elif isinstance(m, Conv2DODE):
                n = m.width * m.width * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                print('Init Conv2DODE') 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                print('Init Linear') 
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  
                print('Init BatchNorm2d')   
        
    def forward(self,x):
        return self.net(x)
    
    def refine(self, variance=0.0):
        new = copy.deepcopy(self)
        new.time_d = 2*self.time_d
        new.scheme = self.scheme
        new.net = nn.Sequential(*[ refine(mod, variance) for mod in self.net])
        return new
