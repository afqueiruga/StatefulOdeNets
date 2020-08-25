import argparse
import driver

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--model', type=str, default='ContinuousNet', metavar='N', help='model architecture (default: "ContinuousNet")')
parser.add_argument('--alpha', type=int, default=16, help="width of the first's segment hidden layer")
parser.add_argument('--widen_factor', type=int, default=1, metavar='E', help='widen factor (default: 1)')
parser.add_argument('--scheme', type=str, default='euler', metavar='N', help='integrator scheme (default: "euler")' )
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='N', help='dataset. Options are "CIFAR10" or "CIFAR100".')
parser.add_argument('--lr', type=float, default=1e-1, metavar='N', help='learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='N', help='weight_decay (default: 5e-4)')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=512, metavar='N', help='batch size  for test set (default: 512)')
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[30, 60, 90], help='decrease learning rate at these epochs')
parser.add_argument('--lr_decay', type=float, default='0.1',  help='learning rate decay factor')
parser.add_argument('--refine', type=int, nargs='+', default=[], help='refine the network at these epochs')
parser.add_argument('--initial_time_d', type=int, default=3, help="initial time refinement--ie, number of layers--of each segment")
parser.add_argument('--n_time_steps_per', type=int, default=1, help="number of time-steps per time_d to take during forward pass")
parser.add_argument('--time_epsilon', type=float, default=None, help="How long is the depth-time")
parser.add_argument('--batch_norm', default=True, help='include batch norm layers', action='store_true')
parser.add_argument('--use_skipinit', default=False, help='use skip init', action='store_true')
parser.add_argument('--seed', type=int, default='1',  help='Seed value')
parser.add_argument('--device', type=str, default=None, help='Which pytorch device?')

args = parser.parse_args()

def drive_by_args(args):
    driver.do_a_train_set(
        args.dataset,
        args.model,
        args.alpha,
	    args.widen_factor,
        args.scheme,
        args.batch_norm,
        args.initial_time_d,
        args.time_epsilon,
        args.n_time_steps_per,
        N_epochs = args.epochs,
        N_adapt = args.refine,
        lr = args.lr,
        lr_decay = args.lr_decay,
        epoch_update = args.lr_decay_epoch,
        use_skip_init = args.use_skipinit,
        weight_decay = args.weight_decay,
        seed = args.seed,
        device = args.device)

drive_by_args(args)
