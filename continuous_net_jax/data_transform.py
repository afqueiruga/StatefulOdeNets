import jax.numpy as jnp


def data_transform(dataset):
    """Reorders torch formatted images to jax shapes and types."""
    for X, Y in dataset:
        yield jnp.array(X).transpose((0, 2, 3, 1)), jnp.array(Y)
