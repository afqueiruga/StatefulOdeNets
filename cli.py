import argparse
import driver

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--model', type=str, default='odenet', metavar='N', help='Model')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='N', help='dataset. Options are "CIFAR10" or "FMIST".')
parser.add_argument('--lr', type=float, default=1e-1, metavar='N', help='learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=1e-5, metavar='N', help='weight_decay (default: 1e-5)')
parser.add_argument('--epochs', type=int, default=110, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--batch', type=int, default=128, metavar='N', help='batch size (default: 10000)')
parser.add_argument('--batch_test', type=int, default=128, metavar='N', help='batch size  for test set (default: 10000)')
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--folder', type=str, default='results_det',  help='specify directory to print results to')
parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 60, 90], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_decay', type=float, default='0.1',  help='PCL penalty lambda hyperparameter')
parser.add_argument('--refine', type=int, nargs='+', default=[], help='Decrease learning rate at these epochs.')

parser.add_argument('--scheme', type=str, default='euler')
parser.add_argument('--alpha', type=int, default=16, help="width of the first's segment hidden layer")
parser.add_argument('--initial_time_d', type=int, default=3, help="initial time refinement--ie, number of layers--of each segment")
parser.add_argument('--n_time_steps_per', type=int, default=1, help="number of time-steps per time_d to take during forward pass")
parser.add_argument('--time_epsilon', type=float, default=1.0, help="How long is the depth-time")
parser.add_argument('--use_batch_norms', default=False, help='include batch norm layers', action='store_true')

parser.add_argument('--use_kaiming', default=False, help='include kaiming', action='store_true')

parser.add_argument('--seed', type=int, default='1',  help='Prediction steps')
parser.add_argument('--device', type=str, default=None, help='Which pytorch device?')

args = parser.parse_args()

print(args.use_batch_norms)

def drive_by_args(args):
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
        use_kaiming=args.use_kaiming,
        weight_decay=args.wd,
        seed=args.seed,
        device=args.device)

drive_by_args(args)