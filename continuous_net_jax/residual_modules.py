from typing import Callable, Tuple

import flax
import flax.linen as nn
import jax


class ShallowNet(nn.Module):
    hidden_dim: int
    output_dim: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim, use_bias=self.use_bias)(x)
        h = nn.tanh(h)
        return nn.tanh(nn.Dense(self.output_dim, use_bias=self.use_bias)(h))


class ResidualUnit(nn.Module):
    hidden_features: int
    # norm: nn.Module = nn.BatchNorm
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.hidden_features, (3,3))(x)
        h = self.activation(h)
        h = nn.Conv(x.shape[-1], (3,3))(h)
        h = self.activation(h)
        return h


class ResidualStitch(nn.Module):
    hidden_features: int
    output_features: int
    # norm: nn.Module = nn.BatchNorm
    activation: Callable = nn.relu
    strides: Tuple[int] = (2, 2)

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.hidden_features, (3,3))(x)
        h = self.activation(h)
        h = nn.Conv(self.output_features, (3,3), strides=self.strides)(h)
        h = self.activation(h)
        x_down = nn.Conv(self.output_features, (1, 1), strides=self.strides)(x)
        return x_down + h
