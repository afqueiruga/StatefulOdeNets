from typing import Any, Iterable, List, Tuple

from flax.training.common_utils import onehot
import jax
import jax.numpy as jnp
import tqdm

Array = Any


class Metrics():
    losses: List[float]

    def __init__(self):
        self.losses = []


def cross_entropy_loss(y_label, logp_y_pred):
    return -jnp.mean(jnp.sum(onehot(y_label, logp_y_pred.shape[-1]) * logp_y_pred, axis=-1))


def decojit(*args, **kwargs):

    def closedjit(f):
        return jax.jit(f, *args, **kwargs)

    return closedjit


class Trainer():
    """This class is basically just a container of closures to jit."""
    train_data: Iterable[Tuple[Array, Array]]
    test_data: Iterable[Tuple[Array, Array]]
    model: Any

    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

        @jax.jit
        def train_step(optimizer, X, Y):
            """Train for a single step using self.model."""
            print('tracing train_step')

            def loss_fn(params):
                logp_y_pred = self.model.apply({'params': params}, X)
                loss = cross_entropy_loss(Y, logp_y_pred)
                return loss, logp_y_pred

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grad = grad_fn(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            return optimizer, loss

        self.train_step = train_step

        @jax.jit
        def test_metrics(params, X, Y):
            print('tracing test_metrics')
            logp_y_pred = self.model.apply({'params': params}, X)
            # loss = cross_entropy_loss(Y, logp_y_pred)
            return jnp.mean(jnp.argmax(logp_y_pred, -1) == Y)

        self.test_metrics = test_metrics

    def train_epoch(self, optimizer):
        """Loop over train_data once, applying the optimizer."""
        metrics = Metrics()
        for X, Y in tqdm.tqdm(self.train_data, desc="Epoch"):
            optimizer, loss = self.train_step(optimizer, X, Y)
            metrics.losses.append(loss)
        return optimizer, metrics

    def metrics_over_test_set(self, params):
        accuracies = []
        for X, Y in self.test_data:
            accuracies.append(self.test_metrics(params, X, Y))
        return jnp.mean(jnp.array(accuracies))
