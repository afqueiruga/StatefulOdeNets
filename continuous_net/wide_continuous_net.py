from torch import nn
import math
from .ode_models import *


def NoSequential(*args):
    """Filters Nones as no-ops when making a nn.Sequential to allow for architecture toggling."""
    net = [ arg for arg in args if arg is not None ]
    return nn.Sequential(*net)


class WideContinuousNet(nn.Module):
    """Uses one OdeBlock per segment."""
    def __init__(self,
                 ALPHA=16,
                 widen_factor=4,
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
                 activation_before_conv=True):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        self.use_batch_norms = use_batch_norms
        self.stitch_epsilon = time_epsilon / (time_d*n_time_steps_per)

        if activation_before_conv:
            _OdeUnit = ShallowConv2DODE_Flipped
            _ODEStitch = ODEStitch_Flipped
        else:
            _OdeUnit = ShallowConv2DODE
            _ODEStitch = ODEStitch

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
            _stitch_macro = lambda _alpha, _beta, stride=2 : \
                _ODEStitch(_alpha, _beta, _beta,
                          epsilon=self.stitch_epsilon,
                          use_batch_norms=use_batch_norms,
                          use_skip_init=use_skip_init,
                          stride=stride)
        else:
            _stitch_macro = lambda _alpha, _beta : \
                nn.Conv2d(_alpha, _beta, kernel_size=1, padding=1, stride=2, bias=False)

        # The full network, with three OdeBlocks (_macro)
        self.net = NoSequential(
            nn.Conv2d(
                in_channels, ALPHA, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(ALPHA) if use_batch_norms else None,
            nn.ReLU(),
            _stitch_macro(ALPHA, ALPHA*widen_factor, stride=1) if widen_factor > 1 else None,
            _macro(ALPHA*widen_factor),
            _stitch_macro(ALPHA*widen_factor, 2*ALPHA*widen_factor),
            _macro(2*ALPHA*widen_factor),
            _stitch_macro(2*ALPHA*widen_factor, 4*ALPHA*widen_factor),
            _macro(4*ALPHA*widen_factor),
            nn.BatchNorm2d(4*ALPHA*widen_factor, momentum=0.9) if activation_before_conv else None,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4*ALPHA*widen_factor,out_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Conv2DODE):
                n = m.width * m.width * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        return self.net(x)

    def refine(self, variance=0.0):
        new = copy.deepcopy(self)
        new.time_d = 2*self.time_d
        new.scheme = self.scheme
        new.net = nn.Sequential(*[ refine(mod, variance) for mod in self.net])
        return new
