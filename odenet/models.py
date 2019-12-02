import torch
import functools

#
# Helper functions for making standard networks
#
def shallow(in_dim,hidden,out_dim,Act=torch.nn.ReLU):
    """Just make a shallow network. This is more of a macro."""
    return torch.nn.Sequential(
            torch.nn.Linear(in_dim,hidden),
            Act(),
            torch.nn.Linear(hidden,out_dim),
        )
def deep(widths,Act=torch.nn.ReLU):
    """Make a deep FCMLP given width specifications. Degenerates to a shallow layer if len(widths)==3"""
    layers = []
    for i in range(len(widths)-2):
        layers.extend([torch.nn.Linear(widths[i],widths[i+1]), Act()])
    layers.append(torch.nn.Linear(widths[-2],widths[-1]))
    return torch.nn.Sequential(*layers)


def channel_squish(imgs, sq):
    """TODO: Only works without channels right now"""
    W,H = imgs.shape[-2],imgs.shape[-1]
    C = imgs.shape[1]
    ii = imgs.reshape(-1, W, H//sq, sq).permute(0,2,1,3).reshape(-1,W//sq, H//sq, sq*sq).permute(0,3,2,1)
    return ii


#
# Extra Building Blocks
#
class MultiLinear(torch.nn.Module):
    """Like Linear, but allows for higher ranks."""
    def __init__(self, in_dims, out_dims, bias=True):
        super(MultiLinear, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        in_features = functools.reduce(lambda x,y:x*y, in_dims)
        out_features = functools.reduce(lambda x,y:x*y, out_dims)
        self.net = torch.nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        xflat = torch.flatten(x, start_dim=-len(self.in_dims))
        hflat = self.net(xflat)
        return torch.reshape( hflat, hflat.shape[:-1]+self.out_dims )

class MultiLinearODE(MultiLinear):
    """MultiLinear with a t"""
    def __init__(self, *args, **kwargs):
        super(MultiLinearODE,self).__init__(*args,**kwargs)
    def forward(self, t, x):
        xflat = torch.flatten(x, start_dim=-len(self.in_dims))
        hflat = self.net(xflat)
        return torch.reshape( hflat, hflat.shape[:-1]+self.out_dims )
    
#
# Basic Classes
#
class ShallowNet(torch.nn.Module):
    """Just a basic shallow network"""
    def __init__(self, in_dim, out_dim, hidden=10, Act=torch.nn.ReLU):
        super(ShallowNet,self).__init__()
        self.net = shallow(in_dim,hidden,out_dim,Act=Act)
    def forward(self,x):
        return self.net(x)
    
class ShallowSkipNet(torch.nn.Module):
    """A basic shallow network with a skip connection"""
    def __init__(self, dim, hidden=10, Act=torch.nn.ReLU):
        super(ShallowSkipNet,self).__init__()
        self.net = shallow(dim,hidden,dim,Act=Act)
    def forward(self,x):
        return x+self.net(x)

class DeepNet(torch.nn.Module):
    """A deep network"""
    def __init__(self, dims, Act=torch.nn.ReLU):
        super(DeepNet,self).__init__()
        self.net = deep(dims,Act=Act)
    def forward(self,x):
        return self.net(x)
