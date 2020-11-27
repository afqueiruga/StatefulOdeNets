from typing import Any, Iterable, Tuple

import jax.numpy as jnp


class DataTransform:
    """Reorders torch formatted images to jax shapes and types."""
    dataset: Iterable[Tuple[Any, Any]]

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for X, Y in self.dataset:
            yield jnp.array(X).transpose((0, 2, 3, 1)), jnp.array(Y)
