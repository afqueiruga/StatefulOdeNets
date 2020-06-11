"""
This script replicates the parameters for RefineNet applied to CIFAR10 and
CIFAR100 to reproduce Table 1 and Figure 4.
"""
from typing import List
import attr

import driver


@attr.s(auto_attribs=True)
class Pack:
    dataset: str
    model: str
    scheme: str
    alpha: int
    initial_time_d: int  # t in [0,1] cut up into time_d slices
        # t=1, t=2, t=3,  -> if resnet epsilon = 1, then
        # t=1/3, t=2/3, t=1.0 -> resnet epsilon = 1/3
    time_epsilon: float = 1.0  # xdot = epsilon * f(x,t)
    use_batch_norms: str = ""
    use_skip_init: bool = False
    
    n_time_steps_per: int = 1
    epochs: int = 160
    lr: float = 0.1
    wd: float = 5e-4
    use_adjoint: bool = False

    lr_decay: float = 0.1
    lr_update: List[int] = None
    refine: List[int] = None
    refine_variance: float = 0.1
    use_kaiming: bool = False
    shape_function: str = 'piecewise'
        
    width: int = None
    seed: int = 1
    

# Results for CIFAR1
# 16 wide, Nt=32 deep
models_cifar10 = [
    ("rk4_classic",1,[40, 50, 60, 70, 80]),
    ("euler",32,[]),
]
experiment_cifar10 = [
    Pack("CIFAR10",
         "RefineNetActFirst",
         scheme,
         16,
         initial_time_d,
         time_epsilon=8.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
          epochs = 150,
         use_adjoint=False,
         seed=seed)
    for scheme,initial_time_d, refine in models_cifar10
    for seed in [1, 2, 3, 4, 5]
]


# Results for CIFAR100
# 4x wide, Nt=8 deep
models_cifar100 = [
    ("rk4_classic",1,[20,50,80]),
    ("euler",8,[]),
]
experiment_cifar100 = [
    Pack("CIFAR100",
         "RefineNetActFirst",
         scheme,
         64,
         initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme, initial_time_d, refine in models_cifar100
    for seed in [1, 2, 3, 4, 5]
]

# Run an "inetgration test"
models_test = [
    ("rk4_classic",1,[1,]),
    ("euler",4,[]),
]
experiment_short_test = [
    Pack("CIFAR10",
         "RefineNetActFirst",
         scheme,
         4,
         initial_time_d,
         time_epsilon=8.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         epochs = 2,
         use_adjoint=False,
         seed=seed)
    for scheme,initial_time_d, refine in models_test
    for seed in [1]
]

DEVICE = "cuda:3"
# Change this to swap experiment
args_list = experiment_short_test
for args in args_list:
    driver.do_a_train_set(
        args.dataset,
        args.model,
        args.alpha,
        args.scheme,
        args.use_batch_norms,
        args.initial_time_d,
        args.time_epsilon,
        args.n_time_steps_per,
        N_epochs=args.epochs,
        N_adapt=args.refine,
        lr=args.lr,
        lr_decay=args.lr_decay,
        epoch_update=args.lr_update,
        weight_decay=args.wd,
        use_adjoint=args.use_adjoint,
        use_kaiming=args.use_kaiming,
        use_skip_init=args.use_skip_init,
        width=args.width,
        refine_variance=args.refine_variance,
        seed=args.seed,
        device=DEVICE)
