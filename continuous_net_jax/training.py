from typing import Any, Callable, Iterable, List, Tuple

from flax.training.common_utils import onehot
import jax
import jax.numpy as jnp
import tqdm

from .continuous_types import *
from .learning_rate_schedule import LearningRateSchedule


def pack_params(params, state):
    return {'params': params, **state}


def cross_entropy_loss(y_label, logp_y_pred):
    return -jnp.mean(
        jnp.sum(onehot(y_label, logp_y_pred.shape[-1]) * logp_y_pred, axis=-1))


class Trainer():
    """This class is basically just a container of closures to jit."""
    train_data: Iterable[Tuple[ArrayType, ArrayType]]
    model: Any

    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data

        @jax.jit
        def train_step(optimizer, state, X, Y, lr):
            """Train for a single step using self.model."""
            print('Tracing train_step.')

            def loss_fn(params):
                logp_y_pred, new_state = self.model.apply(pack_params(
                    params, state),
                                                          X,
                                                          mutable=state.keys())
                loss = cross_entropy_loss(Y, logp_y_pred)
                acc = jnp.mean(jnp.argmax(logp_y_pred, -1) == Y)
                return loss, (acc, new_state)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (acc, new_state)), grad = grad_fn(optimizer.target)
            optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
            return optimizer, new_state, loss, acc

        self.train_step = train_step

    def train_epoch(self, optimizer: Any, state: Any, learning_rate: float,
                    loss_saver: Callable[[Any], None],
                    train_acc_saver: Callable[[Any], None]):
        """Loop over train_data once, applying the optimizer."""
        acc_avg_numerator = 0
        acc_avg_denominator = 0
        for i, (X, Y) in tqdm.tqdm(enumerate(self.train_data), desc="Epoch"):
            optimizer, state, loss, acc = self.train_step(
                optimizer, state, X, Y, learning_rate)
            loss_saver(float(loss))
            train_acc_saver(float(acc))
            acc_avg_numerator += float(acc) * len(Y)
            acc_avg_denominator += len(Y)
        print("Average trian acc ", acc_avg_numerator / acc_avg_denominator)
        return optimizer, state


class Tester():
    """This class is basically just a container of closures to jit."""
    test_data: Iterable[Tuple[ArrayType, ArrayType]]
    model: Any

    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

        @jax.jit
        def test_metrics(params, state, X, Y):
            print('Tracing test_metrics.')
            logp_y_pred, _ = self.model.apply(pack_params(
                params, state),
                                                      X,
                                                      mutable=state.keys())
            # loss = cross_entropy_loss(Y, logp_y_pred)
            return jnp.mean(jnp.argmax(logp_y_pred, -1) == Y)

        self.test_metrics = test_metrics

    def metrics_over_test_set(self, params, state):
        accuracies = []
        denominator = 0
        for X, Y in self.test_data:
            accuracies.append(len(Y) * self.test_metrics(params, state, X, Y))
            denominator += len(Y)
        return jnp.sum(jnp.array(accuracies)) / denominator
