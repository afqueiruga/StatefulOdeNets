from typing import List
import attr
import driver


@attr.s(autto_atribs=True)
class Pack:
    dataset: str
    model: str
    scheme: str
    alpha: int
    initial_time_d: int
    time_epsilon: float
    use_batch_norms: bool

    epochs: int = 20
    # batch_size: int = 128
    # test_batch_size: int = 200

    lr: float = 0.1
    wd: float = 5e-4
    
    n_time_steps:int = 1
    lr_decay: float = 
    lr_update: List[int] = 
    refine: List[int] = None
    
args_list = [
    Pack("FMNIST", "SingleSegment", "rk4",
         8, 3, 0.5, True),
    Pack("FMNIST", "SingleSegment", "rk4",
         8, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "midpoint",
         8, 3, 0.5, True),
    Pack("FMNIST", "SingleSegment", "midpoint",
         8, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "euler",
         8, 3, 0.5, True),
    Pack("FMNIST", "SingleSegment", "euler",
         8, 3, 0.5, False),
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
        device="cuda:0")