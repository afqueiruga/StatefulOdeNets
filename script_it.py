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
        
    width: int = None
    seed: int = 1
    
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
#     (1,[10,20,30]),
#     (1,[20,30,40]),
#     (1,[30,45,60]),
#     (2,[30,40]),
#     (2,[45,60]),
#     (2,[60,120]),
#     (2,[50,100]),
#     (2,[40,80]),
#     (2,[30,60]),
#     (1,[10,20,40,60]),
#     (1,[10,20,30,40,50]),
# The above one failed for rk4 on a memory error
# So, I implemented relu(inplace=True) and was greedier with this:
    (1,[20,40,60,70,80])
]
experiment_1249May17_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False)
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

# Experiment at 1800 May 22
# Wide odenet
# This was the wrong architecture
cuda0_batch = ["euler"]
cuda3_batch = ["rk4_classic"]
refinements_batch = [
    (1,[40,80]),
    (1,[20,50,80]),
    (1,[20, 40, 60, 70, 80])
]
experiment_1800May22_list = [
    Pack("CIFAR10", "Wide", scheme, 16, initial_time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         width=16)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda0_batch
]

# Experiment at 200 May 24
# Wide odenet
cuda0_batch = ["euler"]
cuda3_batch = ["rk4_classic"]
refinements_batch = [
#     (1,[40,80]),
#     (1,[20,50,80]),
#     (1,[20,40,60,70,80])
    (4,[]),
    (8,[])
]
experiment_200May24_list = [
    Pack("CIFAR10", "Wide2", scheme, 16, initial_time_d,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         width=32)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda0_batch
]

# Experiment at 200 May 24
# Wide odenet
cuda0_batch = ["euler"]
cuda3_batch = ["rk4_classic"]
refinements_batch = [
#     (1,[40,80]),
#     (1,[20,50,80]),
#     (1,[20,40,60,70,80])
    (4,[]),
    (2,[]),
    (8,[])
]
experiment_1800May24_list = [
    Pack("CIFAR10", "Wide2", scheme, 16, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         width=64)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda0_batch
]


# Experiment at 200 May 24
# Wide odenet
cuda0_batch = ["euler"]
cuda3_batch = ["rk4_classic"]
refinements_batch = [
#     (1,[40,80]),
#     (1,[20,50,80]),
#     (1,[20,40,60,70,80])
    (4,[]),
    (2,[]),
    (8,[])
]
experiment_1800May24_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 32, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda0_batch
]

# Experiment at 1800 May 24
# Wide odenet
cuda0_batch = ["euler"]
cuda3_batch = ["rk4_classic"]
refinements_batch = [
#     (1,[40,80]),
#     (1,[20,50,80]),
#     (1,[20,40,60,70,80])
    (2,[]),
    (4,[]),
    (8,[])
]
experiment_1800May24_list = [
    Pack("CIFAR10", "Wide2", scheme, 16, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         width=32)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda0_batch
]


# Experiment at 0100 May 25
# Wide odenet
cuda1_batch = ["euler"]
cuda2_batch = ["rk4_classic"]
refinements_batch = [
    (1,[40,70]),
    (1,[20,50,70]),
    (1,[20,35,50,70]),
    # (1,[20,40,50,60,70])
]
experiment_1800May24_list = [
    Pack("CIFAR10", "Wide2", scheme, 16, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         width=32)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda3_batch
]


# Experiment at 1900 May 25
# Wide odenet
cuda1_batch = ["euler"]
cuda2_batch = ["rk4_classic"]
refinements_batch = [
    (32,[])
]
experiment_1900May25_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         time_epsilon=1.0,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False)
    for initial_time_d, refine in refinements_batch
    for scheme in cuda1_batch
]


# Experiment at 1219 May 29
# More seeds
cuda0_batch = [2, 3, 4]
cuda1_batch = [5, 6, 7]
models_batch = [
    #("euler",32,[]),
    ("rk4_classic",1,[20, 40, 60, 70, 80])
]
experiment_1219May29_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         time_epsilon=1.0,
         use_skip_init=False,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme,initial_time_d, refine in models_batch
    for seed in cuda1_batch
]

# Experiment at 0004 May 30
# More seeds
cuda0_batch = [
    ("rk4_classic",1,[40,80]),
]
cuda1_batch = [
    ("euler",4,[]),
]
experiment_0004May30_list = [
    Pack("CIFAR100", "SingleSegment", scheme, alpha, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False)
    for scheme, initial_time_d, refine in cuda0_batch
    for alpha in [32,64]
]


# Experiment at 0014 May 31
# More seeds
cuda0_batch = [
    ("rk4_classic",1,[40,50,60,80]),
]
cuda1_batch = [
    ("euler",16,[]),
]
experiment_0014May31_list = [
    Pack("CIFAR100", "SingleSegment", scheme, alpha, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme, initial_time_d, refine in cuda0_batch
    for alpha in [32]
    for seed in [1,2,3]
]

# Experiment at 0014 May 31
# More seeds
cuda0_batch = [
    ("rk4_classic",1,[20,50,80]),
]
cuda2_batch = [
    ("euler",8,[]),
]
experiment_1538May31_list = [
    Pack("CIFAR100", "SingleSegment", scheme, alpha, initial_time_d,
         time_epsilon=4.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme, initial_time_d, refine in cuda0_batch
    for alpha in [64]
    for seed in [6,7]#[4,]#[1,2,3]
]



# Experiment at 824 May 30
# Changing epsilon cifar10
# epsilon=32, skip=False,True fails
# epsilon=16, skip=False,True fails
# epsilon=8, skip=False, fails
# epsilon=8, skip=True worked
seed_cudaA_batch = [1, 2, 3]
seed_cudaB_batch = [4, 5, 6]
models_cuda1_batch = [
    ("rk4_classic",1,[20, 40, 60, 70, 80])
]
models_cuda2_batch = [
    ("euler",32,[]),
]

experiment_824June3_list = [
    Pack("CIFAR10", "SingleSegment", scheme, 16, initial_time_d,
         time_epsilon=8.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme,initial_time_d, refine in models_cuda2_batch
    for seed in seed_cudaB_batch
]



# Experiment at 1036 Jun 3
# CIFAR100 with the flipped one
seed_batch_A = [1,2,3]
seed_batch_B = [4,5,6]
scheme_batch_0 = [
    ("rk4_classic",1,[20,50,80]),
]
scheme_batch_1 = [
    ("euler",8,[]),
]
experiment_1036Jun3_list = [
    Pack("CIFAR100", "RefineNet", scheme, alpha, initial_time_d,
         time_epsilon=8.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme, initial_time_d, refine in scheme_batch_0
    for alpha in [64]
    for seed in seed_batch_B
]

# Experiment at 824 May 30
# Changing epsilon cifar10
# epsilon=32, skip=False,True fails
# epsilon=16, skip=False,True fails
# epsilon=8, skip=False, fails
# epsilon=8, skip=True worked
seed_batch_A = [1,2,3]
seed_batch_B = [4,5,6]
scheme_batch_0 = [
    ("rk4_classic",1,[20,40,60]),
]
experiment_613June4_list = [
    Pack("CIFAR100", "RefineNet", scheme, alpha, initial_time_d,
         time_epsilon=8.0,
         use_skip_init=True,
         use_batch_norms="ode",
         refine=refine,
         lr=0.1,
         lr_decay=0.1,
         lr_update=[80, 120, 140],
         use_adjoint=False,
         seed=seed)
    for scheme, initial_time_d, refine in scheme_batch_0
    for alpha in [64]
    for seed in seed_batch_B
]


DEVICE = "cuda:0"
args_list = experiment_613June4_list
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
        shape_function=args.shape_function,
        seed=args.seed,
        device=DEVICE)