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
    time_epsilon: float  # xdot = epsilon * f(x,t)
    use_batch_norms: bool
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
    #(2,[40,80]),
    (2,[50,100]),
    (2,[60,120]),
]
cuda1_batch = [
    (1,[10,40,60,80])
]
args_list = [
    Pack("CIFAR10", "SingleSegment", scheme, alpha,
         initial_time_d, epsilon, False,
         refine=refine,
         refine_variance=refine_variance,
         use_skip_init=True,
         n_time_steps_per=n_time_steps_per,
         lr=0.05,
         lr_decay=0.1,
         use_adjoint=use_adjoint,
         lr_update=[80, 120, 140]
        )
    for n_time_steps_per in [1]
    for epsilon in [1.0]
    for initial_time_d, refine in cuda1_batch
    for refine_variance in [0.0]
    for alpha in [16]
    for scheme in [ "rk4_classic", ]
    for use_adjoint in [ False ]
]

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
        device="cuda:1")