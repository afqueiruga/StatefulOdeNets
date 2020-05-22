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
    # batch_size: int = 128
    # test_batch_size: int = 200

    lr: float = 0.1
    wd: float = 5e-4
    use_adjoint: bool = False

    lr_decay: float = 0.1
    lr_update: List[int] = None
    refine: List[int] = None
    refine_variance: float = 0.1
    use_kaiming: bool = False
    shape_function: str = 'piecewise'
    
cuda0_batch = [
    (2,[40,80]),
    (2,[50,100]),
    (2,[60,120]),
]
nothing = [
    Pack("CIFAR10", "SingleSegment", scheme, alpha,
         initial_time_d, epsilon, False,
         refine=refine,
         refine_variance=refine_variance,
         use_skip_init=True,
         n_time_steps_per=n_time_steps_per,
         lr=0.1,
         lr_decay=0.1,
         use_adjoint=use_adjoint,
         lr_update=[80, 120, 140]
        )
    for n_time_steps_per in [1]
    for epsilon in [1.0]
    for initial_time_d, refine in []
    for refine_variance in [0.0]
    for alpha in [16]
    for scheme in [ "rk4_classic", ]
    for use_adjoint in [ False ]
]
reference_test = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140])
    for time_d in [1, 2, 3, 4]
    for scheme in ["rk4_classic"]
]

# Experiment at 12:49 May 17
# Check refinement configurations of Euler vs rk4
cuda1_batch = ["euler"]
cuda2_batch = ["rk4_classic"]
refinements_batch = [
    (1,[10,20,30]),
    (1,[20,30,40]),
    (1,[30,45,60]),
    (2,[30,40]),
    (2,[45,60]),
    (2,[60,120]),
    (2,[50,100]),
    (2,[40,80]),
    (2,[30,60]),
    (1,[10,20,40,60]),
    (1,[10,20,30,40,50]),
]
experiment_1249May17_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140])
    for initial_time_d, refine in refinements_batch
    for scheme in cuda2_batch
]

# Experiment at 1:22 May 17
# Try adjoint with different n_time_steps_per
# Added isnan for this
refinements_batch = [
    (4,[]),
    (8,[]),
    (1,[20,40,]),
    (1,[20,40,60,])
]
n_time_steps_per_batch = [ 1, 2, 3, 4 ]
experiment_122May17_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=True)
    for initial_time_d, refine in refinements_batch
    for scheme in ["euler", "rk4_classic",]
]

# Experiment at 1:22 May 18
# Try adjoint with different n_time_steps_per
# Previous experiment didn't do that >.<
# Added isnan for this
refinements_batch = [
    (4,[]),
    (8,[]),
]
n_time_steps_per_batch = [ 2, 3, 4 ]
experiment_122May18_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         n_time_steps_per=n_time_steps_per,
         use_adjoint=True)
    for initial_time_d, refine in refinements_batch
    for n_time_steps_per in n_time_steps_per_batch
    for scheme in ["euler", "rk4_classic",]
]

# Experiment at 1:16 May 19
# Doing no-refinement manifestation studies with SkipInit no batch norm
cuda1_batch = ["euler"]
cuda2_batch = ["rk4_classic"]
experiment_116May19_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=True,
         use_batch_norms=False,
         refine=[],
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         n_time_steps_per=1,
         use_adjoint=False)
    for initial_time_d in [ 2, 4, 6, 8, 12 ]
    for scheme in cuda2_batch
]

# Experiment at 18:53 May 19
# Refinement with SkipInit and no BatchNorm for Euler/rk4
refinements_batch = [
    (2,[40, 60]),
    (2,[40, 100]),
    (2,[90, 130]),
]
cuda2_batch = ["euler"]
cuda3_batch = ["rk4_classic"]
experiment_1853May19_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=True,
         use_batch_norms=False,
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         n_time_steps_per=1,
         use_adjoint=False)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda3_batch
]

# Experiment at 3:30 May 19
# Try adjoint with different n_time_steps_per
# SkipInit
refinements_batch = [
    (4,[]),
    (8,[]),
]
cuda0_batch = ["euler"]
cuda2_batch = ["rk4_classic"]
n_time_steps_per_batch = [ 1, 2, 3, 4 ]
experiment_330May19_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=True,
         use_batch_norms=False,
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         n_time_steps_per=n_time_steps_per,
         use_adjoint=True)
    for initial_time_d, refine in refinements_batch
    for n_time_steps_per in n_time_steps_per_batch
    for scheme in cuda0_batch
]


DEVICE = "cuda:0"
args_list = experiment_330May19_list
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
        refine_variance=args.refine_variance,
        shape_function=args.shape_function,
        device=DEVICE)