from typing import Any, Iterable, List, Optional

from datetime import datetime

import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from flax import optim
import flax.training.checkpoints
import tensorflow.summary as tf_summary
from matplotlib import pylab as plt
import numpy as np

import tqdm

from continuous_net import datasets
from continuous_net_jax import *
from .baselines import ResNet
from .learning_rate_schedule import LearningRateSchedule
from .optimizer_factory import make_optimizer
from .tensorboard_writer import TensorboardWriter
from .tools import count_parameters

from .continuous_models import *

from .convergence import project_continuous_net
from .basis_functions import *

_CHECKPOINT_FREQ = 20


def report_count(params, state):
    n_params = count_parameters(params)
    n_state = count_parameters(state)
    print("Model has ", n_params, " params + ", n_state, " state params (",
          n_params + n_state, " total).")


def run_an_experiment(dataset_name: Optional[str] = None,
                      train_data: Optional[Any] = None,
                      validation_data: Optional[Any] = None,
                      test_data: Optional[Any] = None,
                      save_dir: str = '../runs/',
                      dataset_dir: str = '../',
                      which_model: str = 'Continuous',
                      seed: int = 0,
                      alpha: int = 8,
                      hidden: int = 8,
                      n_step: int = 3,
                      scheme: str = 'Euler',
                      n_basis: int = 3,
                      basis: str = 'piecewise_constant',
                      norm: str = 'None',
                      kernel_init: str = 'kaiming_out',
                      which_optimizer: str = 'Momentum',
                      learning_rate: float = 0.1,
                      learning_rate_decay: float = 0.1,
                      learning_rate_decay_epochs: Optional[List[int]] = None,
                      weight_decay: float = 5.0e-4,
                      n_epoch: int = 15,
                      refine_epochs: Optional[Iterable] = None,
                      project_epochs: Optional[Iterable] = None,
                      ):
    
    if dataset_name:
        torch_train_data, torch_validation_data, torch_test_data = (
            datasets.get_dataset(dataset_name, root=dataset_dir))
        train_data = DataTransform(torch_train_data)
        validation_data = DataTransform(torch_validation_data)
        test_data = DataTransform(torch_test_data)
    
    lr_schedule = LearningRateSchedule(learning_rate, learning_rate_decay,
                                       learning_rate_decay_epochs)
    optimizer_def = make_optimizer(which_optimizer,
                                   learning_rate=learning_rate,
                                   weight_decay=weight_decay)

    if refine_epochs == None:
        refine_epochs = []

    if which_model == 'Continuous':
        model = ContinuousImageClassifier(alpha=alpha,
                                          hidden=hidden,
                                          n_step=n_step,
                                          scheme=scheme,
                                          n_basis=n_basis,
                                          basis=basis,
                                          norm=norm)
    elif which_model == 'Continuous2':   
        model = ContinuousImageClassifierSmall(alpha=alpha,
                                          hidden=hidden,
                                          n_step=n_step,
                                          scheme=scheme,
                                          n_basis=n_basis,
                                          basis=basis,
                                          norm=norm)        
        
    elif which_model == 'ResNet':
        model = ResNet(alpha=alpha,
                       hidden=hidden,
                       n_step=n_step,
                       norm=norm,
                       kernel_init=kernel_init)
    else:
        raise ArgumentError("Unknown model class.")
    eval_model = model.clone(training=False)

    # Create savers management.
    exp = Experiment(model, path=save_dir)
    exp.save_optimizer_hyper_params(optimizer_def, seed,
                                    extra={'learning_rate_decay_epochs': learning_rate_decay_epochs,
                              
                                
                                'refine_epochs': refine_epochs})
    tb_writer = TensorboardWriter(exp.path)
    loss_writer = tb_writer.Writer('loss')
    test_acc_writer = tb_writer.Writer('test_accuracy')
    train_acc_writer = tb_writer.Writer('train_accuracy')
    validation_acc_writer = tb_writer.Writer('validation_accuracy')

    # Initialize the model and make training modules.
    prng_key = jax.random.PRNGKey(seed)
    x, _ = next(iter(train_data))
    init_vars = exp.model.init(prng_key, x)
    init_state, init_params = init_vars.pop('params')
    optimizer = optimizer_def.create(init_params)
    current_state = init_state
    report_count(init_params, init_state)
    trainer = Trainer(exp.model, train_data)
    validator = Tester(eval_model, validation_data)
    tester = Tester(eval_model, test_data)
    
    
    print('**** Setup ****')
    n_params = jax.tree_util.tree_reduce(lambda x, y : x + y.flatten().size, init_params, initializer=0)
    print(n_params)
    print('Total params: %.2fk ; %.2fM' % (n_params * 10**-3, n_params * 10**-6))
    print('************')    
    
    
    
    validation_acc = validator.metrics_over_test_set(optimizer.target, current_state)
    test_acc = tester.metrics_over_test_set(optimizer.target, current_state)
    best_test_acc = 0.0
    validation_acc_writer(float(validation_acc))
    print("Initial acc ", test_acc)
    # jax.profiler.save_device_memory_profile(f"{exp.path}/memory_init.prof")
    for epoch in range(1, 1 + n_epoch):
        if epoch in refine_epochs:
            new_model, new_params, current_state = exp.model.refine(
                optimizer.target, current_state)
            exp.model = new_model
            eval_model = exp.model.clone(training=False)
            # We just reset the momenta.
            optimizer = optimizer_def.create(new_params)
            trainer = Trainer(exp.model, train_data)
            validator = Tester(eval_model, validation_data)
            tester = Tester(eval_model, test_data)
            print("Refining model to: ", end='')
            report_count(new_params, current_state)
            print('N basis function: ', exp.model.n_basis)
            print('N Steps: ', exp.model.n_step)    
            
        if epoch in project_epochs:
            
            print('Before N basis function: ', exp.model.n_basis)
            print('Before N Steps: ', exp.model.n_step) 
            
            print("all good")

            new_params, current_state = project_continuous_net(optimizer.target, current_state,
                                            BASIS[exp.model.basis],
                                            BASIS[exp.model.basis], int(exp.model.n_basis/2))
            
            print('Before N basis function: ', exp.model.n_basis)
            print('Before N Steps: ', exp.model.n_step)               
            
            new_model = exp.model.clone(basis=exp.model.basis, n_basis=int(exp.model.n_basis/2))
            new_model = new_model.clone(n_step=int(exp.model.n_step/2), scheme=exp.model.scheme)
        
    
            exp.model = new_model

            print('After N basis function: ', exp.model.n_basis)
            print('After N Steps: ', exp.model.n_step)               
            
            eval_model = exp.model.clone(training=False)
            # We just reset the momenta.
            optimizer = optimizer_def.create(new_params)
            trainer = Trainer(exp.model, train_data)
            validator = Tester(eval_model, validation_data)
            tester = Tester(eval_model, test_data)
            print("Project model to: ", end='')
            report_count(new_params, current_state)
                        
            
            
        optimizer, current_state = trainer.train_epoch(optimizer, current_state,
                                                       lr_schedule(epoch),
                                                       loss_writer,
                                                       train_acc_writer)
        # jax.profiler.save_device_memory_profile(f"{exp.path}/memory_train_{epoch}.prof")
        validation_acc = validator.metrics_over_test_set(optimizer.target, current_state)

        # jax.profiler.save_device_memory_profile(f"{exp.path}/memory_test_{epoch}.prof")
        validation_acc_writer(float(validation_acc))
        print("After epoch ", epoch, "test acc: ", validation_acc)
        if best_test_acc < validation_acc:
            best_test_acc = validation_acc
            exp.save_checkpoint(optimizer, current_state, epoch)
        
        #if epoch % _CHECKPOINT_FREQ == 0:
        #    exp.save_checkpoint(optimizer, current_state, epoch)
        tb_writer.flush()

    #try:  # Save the last checkpoint if the last loop didn't.
    #    exp.save_checkpoint(optimizer, current_state, epoch)
    #except:
    #    pass
    test_acc = tester.metrics_over_test_set(optimizer.target, current_state)
    print("Final test set accuracy: ", test_acc)
    print("Final best test set accuracy: ", best_test_acc)
    
    return test_acc  # Return the final test set accuracy for testing.
