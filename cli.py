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

parser.add_argument('--method', type=str, default='euler')
parser.add_argument('--alpha', type=int, default=16, help='inital number of layers per segment')

parser.add_argument('--seed', type=int, default='1',  help='Prediction steps')
parser.add_argument('--device', type=str, default=None, help='Which pytorch device?')

args = parser.parse_args()


def drive_by_args(args):
    driver.do_a_train_set(
        args.dataset, args.alpha, args.method, N_epochs=args.epochs, N_adapt=args.refine,
        lr=args.lr, lr_decay=args.lr_decay,
        epoch_update=args.lr_update, weight_decay=args.wd,
        seed=args.seed, device=args.device)

drive_by_args(args)