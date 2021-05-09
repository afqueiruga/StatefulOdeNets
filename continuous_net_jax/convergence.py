from typing import Any, Iterable, Tuple

from SimDataDB import SimDataDB2
import functools
import glob
import os

import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp

from continuous_net_jax import *
from continuous_net import datasets
from continuous_net_jax.basis_functions import *


def project_continuous_net(params: Iterable[JaxTreeType],
                               state: Iterable[JaxTreeType], 
                               source_basis: ContinuousParameters, 
                               target_basis: ContinuousParameters,
                               n_basis: int) -> Tuple[Iterable[JaxTreeType],Iterable[JaxTreeType]]:
    PROJ = lambda w_: function_project_tree(w_, source_basis, target_basis,
                                            n_basis)
    p2 = flax.core.unfreeze(params).copy()
    s2 = flax.core.unfreeze(state).copy()
    p2['ContinuousNet_0']['ode_params'] = PROJ(
        params['ContinuousNet_0']['ode_params'])
    p2['ContinuousNet_1']['ode_params'] = PROJ(
        params['ContinuousNet_1']['ode_params'])
    p2['ContinuousNet_2']['ode_params'] = PROJ(
        params['ContinuousNet_2']['ode_params'])

    s2['ode_state']['ContinuousNet_0']['state'] = PROJ(
        state['ode_state']['ContinuousNet_0']['state'])
    s2['ode_state']['ContinuousNet_1']['state'] = PROJ(
        state['ode_state']['ContinuousNet_1']['state'])
    s2['ode_state']['ContinuousNet_2']['state'] = PROJ(
        state['ode_state']['ContinuousNet_2']['state'])

    print('Originally: ', count_parameters(params))
    print('Projected: ', count_parameters(p2))
    return flax.core.freeze(p2), flax.core.freeze(s2)


def interpolate_continuous_net(params: Iterable[JaxTreeType],
                               state: Iterable[JaxTreeType], 
                               source_basis: ContinuousParameters, 
                               target_basis: str,
                               n_basis: int) -> Tuple[Iterable[JaxTreeType],Iterable[JaxTreeType]]:
    INTERP = lambda w_: INTERPOLATE[target_basis](source_basis(w_),
                                                  n_basis)
    p2 = flax.core.unfreeze(params).copy()
    s2 = flax.core.unfreeze(state).copy()
    p2['ContinuousNet_0']['ode_params'] = INTERP(
        params['ContinuousNet_0']['ode_params'])
    p2['ContinuousNet_1']['ode_params'] = INTERP(
        params['ContinuousNet_1']['ode_params'])
    p2['ContinuousNet_2']['ode_params'] = INTERP(
        params['ContinuousNet_2']['ode_params'])

    s2['ode_state']['ContinuousNet_0']['state'] = INTERP(
        state['ode_state']['ContinuousNet_0']['state'])
    s2['ode_state']['ContinuousNet_1']['state'] = INTERP(
        state['ode_state']['ContinuousNet_1']['state'])
    s2['ode_state']['ContinuousNet_2']['state'] = INTERP(
        state['ode_state']['ContinuousNet_2']['state'])

    print('Originally: ', count_parameters(params))
    print('Interpolate: ', count_parameters(p2))
    return flax.core.freeze(p2), flax.core.freeze(s2)


class ConvergenceTester:

    def __init__(self, path: str):
        self.path = path

        exp = Experiment(path=path, scope=globals())
        # The model was saved at the begining, got longer after refinement.
        final_n_step = exp.model.n_step * 2**len(exp.extra['refine_epochs'])
        final_model = exp.model.clone(n_step=final_n_step, n_basis=final_n_step)
        # Load the parameters
        chp = checkpoints.restore_checkpoint(path, None)
        params = chp['optimizer']['target']
        state = chp['state']
        # Initialize a skeleton with the right shape.
        prng_key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 32, 32, 3), jnp.float32)
        p = final_model.init(prng_key, x)
        i_state, i_params = p.pop('params')
        # Reshape the values that were loaded. This is needed because
        # ContinuousNet uses lists in the parameter trees, but the
        # checkpoint always loads dictionaries. I.e., we turn
        # {'ContinuousNet0':{'0':W0, '1':W1}} into
        # {'ContinuousNet0':[W0, W1]}
        loaded_params = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(i_params),
            jax.tree_util.tree_leaves(chp['optimizer']['target']))
        loaded_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(i_state),
            jax.tree_util.tree_leaves(chp['state']))
        eval_model = final_model.clone(training=False)

        self.exp = exp
        self.params = loaded_params
        self.state = loaded_state
        self.eval_model = eval_model

    def perform_convergence_test(self, test_data: Any,
                                 n_steps: Iterable[int],
                                 schemes: Iterable[str]):

        @SimDataDB2(os.path.join(self.path, "convergence.sqlite"))
        def infer_test_error(scheme: str, n_step: int) -> Tuple[float]:
            model = self.eval_model.clone(n_step=n_step, scheme=scheme)
            tester = Tester(model, test_data)
            err = tester.metrics_over_test_set(self.params, self.state)
            return float(err),

        errors = []
        for n_step in n_steps:
            for scheme in schemes:
                error = infer_test_error(scheme, n_step)
                errors.append((n_step, error))
        return errors

    @functools.lru_cache()
    def project(self, target_basis: str, n_basis: int):
        W2, S2 = project_continuous_net(self.params, self.state,
                                        BASIS[self.eval_model.basis],
                                        BASIS[target_basis], n_basis)
        new_model = self.eval_model.clone(basis=target_basis, n_basis=n_basis)
        return new_model, W2, S2

    def perform_project_and_infer(self, test_data: Any,
                                  bases: Iterable[str],
                                  n_bases: Iterable[int],
                                  schemes: Iterable[str],
                                  n_steps: Iterable[int]):
        
        @SimDataDB2(os.path.join(self.path, "convergence.sqlite"))
        def infer_projected_test_error3(scheme: str, n_step: int, basis: str,
                                       n_basis: int) -> Tuple[float, int]:
            # Rely on the LRU cache to avoid the second call, and sqlite 
            # cache to avoid the first call.
            p_model, p_params, p_state = self.project(basis, n_basis)
            s_p_model = p_model.clone(n_step=n_step, scheme=scheme)
            tester = Tester(s_p_model, test_data)
            err = tester.metrics_over_test_set(p_params,  p_state)
            return float(err), count_parameters(p_params)

        print("| Basis | n_basis | Scheme | n_step | error | n_params |")
        print("|-------|----------------------------------------------|")
        errors = {}
        for basis in bases:
            for n_basis in n_bases:
                for n_step in n_steps:
                    for scheme in schemes:
                        e, num_params = infer_projected_test_error3(scheme, n_step, basis, n_basis)
                        print(f"| {basis:20} | {n_basis} | {scheme:5} | {n_step} | {e:1.3f} | {num_params} |")

    @functools.lru_cache()
    def interpolate(self, target_basis, n_basis):
        W2, S2 = interpolate_continuous_net(self.params, self.state,
                                        BASIS[self.eval_model.basis],
                                        target_basis, n_basis)
        new_model = self.eval_model.clone(basis=target_basis, n_basis=n_basis)
        return new_model, W2, S2
    
    def perform_interpolate_and_infer(self, test_data: Any,
                                  bases: Iterable[str],
                                  n_bases: Iterable[int],
                                  schemes: Iterable[str],
                                  n_steps: Iterable[int]):
        
        @SimDataDB2(os.path.join(self.path, "convergence.sqlite"))
        def infer_interpolated_test_error2(scheme: str, n_step: int, basis: str,
                                       n_basis: int) -> Tuple[float, int]:
            # Rely on the LRU cache to avoid the second call, and sqlite 
            # cache to avoid the first call.
            p_model, p_params, p_state = self.interpolate(basis, n_basis)
            s_p_model = p_model.clone(n_step=n_step, scheme=scheme)
            tester = Tester(s_p_model, test_data)
            err = tester.metrics_over_test_set(p_params,  p_state)
            return float(err), count_parameters(p_params)

        print("| Basis | n_basis | Scheme | n_step | error | n_params |")
        print("|-------|----------------------------------------------|")
        errors = {}
        for basis in bases:
            for n_basis in n_bases:
                for n_step in n_steps:
                    for scheme in schemes:
                        e, num_params = infer_interpolated_test_error2(scheme, n_step, basis, n_basis)
                        print(f"| {basis:20} | {n_basis} | {scheme:5} | {n_step} | {e:1.3f} | {num_params} |")

