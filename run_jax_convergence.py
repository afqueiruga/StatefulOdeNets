import glob
import os

from continuous_net_jax.convergence import *
from continuous_net_jax.basis_functions_plotting import *
from continuous_net_jax.tools import *

DIR = "../runs_cifar10_b/"
paths = glob.glob(f"{DIR}/*")

dataset_dir='..'
torch_train_data, torch_validation_data, torch_test_data = (
            datasets.get_dataset('CIFAR10', root=dataset_dir))
train_data = DataTransform(torch_train_data)
validation_data = DataTransform(torch_validation_data)
test_data = DataTransform(torch_test_data)

for path in paths:
    ct = ConvergenceTester(path)
    print(ct.eval_model)
    ct.perform_interpolate_and_infer(
        test_data,
                             ('piecewise_constant', 'fem_linear'),
                            range(1, ct.eval_model.n_step+1), 
                            ['Euler', 'RK4'],
                            (ct.eval_model.n_step,))
#     ct.perform_project_and_infer(test_data,
#                              ('piecewise_constant', 'fem_linear'),
#                             range(1, ct.eval_model.n_step+1), 
#                             ['Euler', 'RK4'],
#                             (ct.eval_model.n_step,))
    #ct.perform_convergence_test(test_data, range(1,16),
    #                            ['Euler','Midpoint','RK4','RK4_38'])