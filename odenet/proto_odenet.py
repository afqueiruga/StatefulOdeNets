"""
the first prototype for how odenets would split. here for posterity
"""

class ODEBlock(torch.nn.Module):
    """Wraps an ode-model with the odesolve to fit into standard 
    models."""
    def __init__(self,net,t_max=1.0,method='euler'):
        super(ODEBlock,self).__init__()
        self.t_max = t_max
        self.method = method
        self.ts = torch.tensor([0,t_max])
        self.net = net
    def forward(self,x):
        h = torchdiffeq.odeint(self.net, x, self.ts,
                               method=self.method)[1,:,:]
        return h
    def refine(self):
        """ddCut self.net in half"""
        front_net = copy.deepcopy(self.net)
        back_net = copy.deepcopy(self.net)
        return torch.nn.Sequential(
            ODEBlock(front_net, self.t_max/2.0),
            ODEBlock(back_net,  self.t_max/2.0),
        )

class ODEModel(torch.nn.Module):
    def __init__(self,i_dim,o_dim,ode_width=4,
                 inside_width=4,Act=torch.nn.ReLU,
                method='euler'):
        super(ODEModel,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(i_dim,ode_width),
            ODEBlock(
                ShallowODE(ode_width,hidden=inside_width,
                           Act=Act),
                method='euler'),
            torch.nn.Linear(ode_width,o_dim),
        )i wi
    def forward(self,x):
        # Missing sigmoid
        y = self.net(x)
        return y
    def refine(self):
        new = copy.deepcopy(self)
        new.net[1] = refine(self.net[1])
        return new
    

