"""Transformer-based langauge models."""

from typing import Callable, Any, Optional

from flax import linen as nn
from flax import struct

import jax.numpy as jnp
import numpy as np

from continuous_net_jax import *
from .baseline_models import *


class dxdtEncoder1DBlock(nn.Module):
    """Transformer encoder as an ODE rate.

    dx/dt = SA(x) + MLP(x+SA(x))

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig
    deterministic: bool

    @nn.compact
    def __call__(self, inputs, *, rng=None):
        """Applies Encoder1DBlock module.

        Args:
          inputs: input data.
          deterministic: if true dropout is applied otherwise not.

        Returns:
          output after transformer encoder block.
        """
        cfg = self.config
        deterministic = self.deterministic

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=cfg.get_dtype())(inputs)
        SA = nn.SelfAttention(num_heads=cfg.num_heads,
                           dtype=cfg.get_dtype(),
                           qkv_features=cfg.qkv_dim,
                           kernel_init=cfg.get_kernel_init(),
                           bias_init=cfg.get_bias_init(),
                           use_bias=False,
                           broadcast_dropout=False,
                           dropout_rate=cfg.attention_dropout_rate,
                           deterministic=True)
        x = SA(x)
        h_self_attention = nn.Dropout(rate=cfg.dropout_rate)(
            x, deterministic=(False if rng is not None else True), rng=rng)
        inner_skip = h_self_attention + inputs

        # MLP block.
        h_mlp = nn.LayerNorm(dtype=cfg.get_dtype())(inner_skip)
        h_mlp = MlpBlock(config=cfg)(h_mlp, deterministic=True)
        return h_mlp + h_self_attention


class ContinuousTransformer(nn.Module):
    """Continuous-Depth Transformer for sequence tagging."""

    config: TransformerConfig
    scheme: str = "Euler"
    n_step: int = 1
    basis: str = "piecewise_constant"
    n_basis: int = 1
    training: bool = True


    @nn.compact
    def __call__(self, *, inputs, train, rng=None):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          train: if it is training.
          rng: the RNG to plumb into dropouts.

        Returns:
          output of a transformer encoder.
        """
        padding_mask = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)[...,
                                                                       None]
        assert inputs.ndim == 2  # (batch, len)
        cfg = self.config

        x = inputs.astype('int32')
        x = nn.Embed(num_embeddings=cfg.vocab_size,
                     features=cfg.emb_dim,
                     name='embed')(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=True)
        x = AddPositionEmbs(cfg)(x)

        dxdt = dxdtEncoder1DBlock(cfg, deterministic=True)
        # Plumb in the rng for dropouts.
        x = ContinuousBlock(dxdt,
                            scheme=self.scheme,
                            n_step=self.n_step,
                            basis=self.basis,
                            n_basis=self.n_basis)(x)
        # Extract the attention matrix as a stateful side-effect to make Fig 1:
        # x = ContinuousBlockSow(
        #         dxdt,
        #         scheme=self.scheme,
        #         n_step=self.n_step,
        #         basis=self.basis,
        #         n_basis=self.n_basis,
        #         name="ContinuousBlock_0")(x)

        x = nn.LayerNorm(dtype=cfg.dtype)(x)
        logits = nn.Dense(cfg.output_vocab_size,
                          kernel_init=cfg.get_kernel_init(),
                          bias_init=cfg.get_bias_init())(x)
        return logits

    def refine(self, params):
        return refine(self, params)
