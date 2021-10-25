"""A trivial linear classifier factorizer to baseline your transformer."""
from typing import Callable, Any, Optional

from flax import linen as nn
from flax import struct

import jax.numpy as jnp
import numpy as np

from continuous_net_jax import *
from .baseline_models import *


class LinearClassifer(nn.Module):
    """Transformer Model for sequence tagging."""

    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, train, rng):
        """Classify Embeddings"""
        cfg = self.config
        x = inputs.astype('int32')
        x = nn.Embed(num_embeddings=cfg.vocab_size,
                     features=cfg.emb_dim,
                     name='embed')(x)
        x = AddPositionEmbs(cfg)(x)
        logits = nn.Dense(cfg.output_vocab_size,
                          kernel_init=cfg.get_kernel_init(),
                          bias_init=cfg.get_bias_init())(x)
        return logits
