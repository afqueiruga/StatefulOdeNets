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


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  dtype: str = 'float32'
  emb_dim: int = 128
  num_heads: int = 1
  num_layers: int = 1
  qkv_dim: int = 128
  mlp_dim: int = 128
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.3
  kernel_init: str = 'xavier_uniform'
  bias_init: str = 'normal'
  posemb_init: str = 'None'

  def __str__(self):
    return f"TransformerConfig_{self.emb_dim},{self.num_heads},{self.num_layers},{self.qkv_dim},{self.mlp_dim}_"

  def get_kernel_init(self) -> Callable:
    return nn.initializers.xavier_uniform()

  def get_bias_init(self) -> Callable:
    return nn.initializers.normal(stddev=1e-6)
    
  def get_posemb_init(self) -> Callable:
    return None

  def get_dtype(self) -> Any:
    return jnp.float32


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.get_posemb_init() is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.get_posemb_init(),
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, deterministic=True):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(cfg.mlp_dim,
                 dtype=cfg.get_dtype(),
                 kernel_init=cfg.get_kernel_init(),
                 bias_init=cfg.get_bias_init())(inputs)
    x = nn.elu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(actual_out_dim,
                      dtype=cfg.get_dtype(),
                      kernel_init=cfg.get_kernel_init(),
                      bias_init=cfg.get_bias_init())(x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      deterministic: if true dropout is applied otherwise not.

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.get_dtype())(inputs)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.get_dtype(),
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.get_kernel_init(),
        bias_init=cfg.get_bias_init(),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=deterministic)(x)

    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.get_dtype())(x)
    y = MlpBlock(config=cfg)(y, deterministic=deterministic)
    return x + y


class Transformer(nn.Module):
  """Transformer Model for sequence tagging."""

  config: TransformerConfig

  @nn.compact
  def __call__(self, *, inputs, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      train: if it is training.

    Returns:
      output of a transformer encoder.

    """
    padding_mask = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)[..., None]
    assert inputs.ndim == 2  # (batch, len)

    cfg = self.config

    x = inputs.astype('int32')
    x = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, name='embed')(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
    x = AddPositionEmbs(cfg)(x)

    for _ in range(cfg.num_layers):
      x = Encoder1DBlock(cfg)(x, deterministic=not train)

    x = nn.LayerNorm(dtype=cfg.get_dtype())(x)
    logits = nn.Dense(
        cfg.output_vocab_size,
        kernel_init=cfg.get_kernel_init(),
        bias_init=cfg.get_bias_init())(x)
    return logits
