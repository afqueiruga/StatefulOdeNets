# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig
    deterministic: bool

    @nn.compact
    def __call__(self, inputs):
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
        x = nn.SelfAttention(num_heads=cfg.num_heads,
                             dtype=cfg.get_dtype(),
                             qkv_features=cfg.qkv_dim,
                             kernel_init=cfg.get_kernel_init(),
                             bias_init=cfg.get_bias_init(),
                             use_bias=False,
                             broadcast_dropout=False,
                             dropout_rate=cfg.attention_dropout_rate,
                             deterministic=deterministic)(x)
        h_self_attention = nn.Dropout(rate=cfg.dropout_rate)(
            x, deterministic=deterministic)
        inner_skip = h_self_attention + inputs

        # MLP block.
        h_mlp = nn.LayerNorm(dtype=cfg.get_dtype())(inner_skip)
        h_mlp = MlpBlock(config=cfg)(h_mlp, deterministic=deterministic)
        return h_mlp + h_self_attention


class ContinuousTransformer(nn.Module):
    """Transformer Model for sequence tagging."""

    config: TransformerConfig
    scheme: str = "Euler"
    n_step: int = 1
    basis: str = "piecewise_constant"
    n_basis: int = 1
    training: bool = True
        
    def __str__(self):
        return f"ContinuousTransformer({self.basis},{self.scheme},{self.config.num_layers},{self.config.emb_dim},{self.config.num_heads},{self.config.qkv_dim},{self.config.mlp_dim})"

    @nn.compact
    def __call__(self, *, inputs, train):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          train: if it is training.

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
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        x = AddPositionEmbs(cfg)(x)

        dxdt = dxdtEncoder1DBlock(cfg, deterministic=True)
        x = ContinuousNetNoState(dxdt,
                          scheme=self.scheme,
                          n_step=self.n_step,
                          basis=self.basis,
                          n_basis=self.n_basis)(x)
        #for _ in range(cfg.num_layers):
        #    x = x + dxdtEncoder1DBlock(cfg)(x, deterministic=not train)

        x = nn.LayerNorm(dtype=cfg.dtype)(x)
        logits = nn.Dense(cfg.output_vocab_size,
                          kernel_init=cfg.get_kernel_init(),
                          bias_init=cfg.get_bias_init())(x)
        return logits

    def refine(self, params):
        return refine(self, params)