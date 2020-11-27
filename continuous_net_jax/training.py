from typing import List

import jax
import jax.numpy as jnp
import tqdm


class Metrics():
    losses: List[float]

    def __init__(self):
        self.losses = []


def onehot(labels, num_classes=10):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    return x.astype(jnp.float32)


def cross_entropy_loss(y_label, logp_y_pred):
    return -jnp.mean(jnp.sum(onehot(y_label) * logp_y_pred, axis=-1))


def train_epoch(model, optimizer, train_data):

    @jax.jit
    def train_step(optimizer, X, Y):
        """Train for a single step."""
        print('tracing')
        def loss_fn(params):
            logp_y_pred = model.apply({'params': params}, X)
            loss = cross_entropy_loss(Y, logp_y_pred)
            return loss, logp_y_pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grad = grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        return optimizer, loss

    metrics = Metrics()
    for X, Y in tqdm.tqdm(train_data):
        optimizer, loss = train_step(optimizer, X, Y)
        metrics.losses.append(loss)
    return optimizer, metrics



def test_metrics(model, params, test_data):
    accuracies = []
    for X, Y in test_data:
        logp_y_pred = model.apply({'params': params}, X)
        loss = cross_entropy_loss(Y, logp_y_pred)
        accuracies = jnp.mean(jnp.argmax(logp_y_pred, -1) == Y)
    return jnp.mean(accuracies)
