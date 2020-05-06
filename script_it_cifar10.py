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
    time_epsilon: float  # xdot = epsilon * f(x,t)
    use_batch_norms: bool

    n_time_steps_per: int = 1
    epochs: int = 99
    # batch_size: int = 128
    # test_batch_size: int = 200

    lr: float = 0.1
    wd: float = 1e-5
    use_adjoint: bool = False
    
    lr_decay: float = 0.1
    lr_update: List[int] = [30,60,90]
    refine: List[int] = None
    use_kaiming: bool = True
    
args_list = [
    Pack("CIFAR10", "SingleSegment", "euler", 16, 3, 1.0, True),
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
        device="cuda:0")