from functools import partial
from typing import Callable, Tuple

import flax
import flax.linen as nn
import jax

import jax.numpy as jnp


xavier_uniform_gain = partial(jax.nn.initializers.variance_scaling, 2, "fan_avg", "uniform")


# Kaiming, but scale by output channel count.
kaiming_out = partial(jax.nn.initializers.variance_scaling, 2.0, "fan_out",
                      "truncated_normal")


# We use string qualifiers instead of function refs.
INITS = {
    'kaiming': jax.nn.initializers.kaiming_normal(),
    'kaiming_out': kaiming_out(),
    'xavier': xavier_uniform_gain(),
    'lecun': jax.nn.initializers.lecun_normal(),
    'glorot': jax.nn.initializers.glorot_uniform(),
    'he': jax.nn.initializers.he_uniform(),
}

NORMS = {
    'None':
        lambda **kwargs: (lambda x: x),
    'BatchNorm':
        nn.BatchNorm,
    'BatchNorm-opt-flax':
        partial(nn.BatchNorm,
                use_running_average=True,
                momentum=0.1,
                epsilon=1e-5),
    'BatchNorm-freeze':
        partial(nn.BatchNorm, use_running_average=True)
}


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
    norm: str = 'BatchNorm'
    activation: Callable = nn.relu
    kernel_init: str = 'kaiming_out'
    training: bool = True
    use_bias: bool = False
    epsilon: float = 1.0

    @nn.compact
    def __call__(self, x):
        h = NORMS[self.norm](use_running_average=not self.training)(x)
        h = self.activation(h)
        h = nn.Conv(self.hidden_features, (3, 3), use_bias=self.use_bias,
                    kernel_init=INITS[self.kernel_init])(h)

        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = self.activation(h)
        h = nn.Conv(x.shape[-1], (3, 3), use_bias=self.use_bias,
                    kernel_init=INITS[self.kernel_init])(h)
        return self.epsilon * h


class ResidualUnitv2(nn.Module):
    hidden_features: int
    norm: str = 'BatchNorm'
    activation: Callable = nn.relu
    kernel_init: str = 'kaiming_out'
    training: bool = True
    use_bias: bool = False
    epsilon: float = 1.0

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.hidden_features, (3, 3), use_bias=self.use_bias,
                    kernel_init=INITS[self.kernel_init])(x)
        h = self.activation(h)
        h = NORMS[self.norm](use_running_average=not self.training)(h)

        h = nn.Conv(x.shape[-1], (3, 3), use_bias=self.use_bias,
                    kernel_init=INITS[self.kernel_init])(h)
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        #h = self.activation(h)
        return self.epsilon * h




class ResidualStitch(nn.Module):
    hidden_features: int
    output_features: int
    norm: str = 'BatchNorm'
    activation: Callable = nn.relu
    kernel_init: str = 'kaiming_out'
    training: bool = True
    strides: Tuple[int] = (2, 2)
    use_bias: bool = False
    epsilon: float = 1.0


    @nn.compact
    def __call__(self, x):
        h = NORMS[self.norm](use_running_average=not self.training)(x)
        h = self.activation(h)
        h = nn.Conv(self.output_features, (3, 3), use_bias=self.use_bias,
                    strides=self.strides, kernel_init=INITS[self.kernel_init])(h)  
        
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = self.activation(h)
        h = nn.Conv(self.output_features, (3, 3), use_bias=self.use_bias,
                    kernel_init=INITS[self.kernel_init])(h)
        
        if self.strides[0] != 1:
            x_down = nn.Conv(self.output_features, (1, 1), use_bias=self.use_bias,
                             strides=self.strides, kernel_init=INITS[self.kernel_init])(x)
            #x_down = NORMS[self.norm](use_running_average=not self.training)(x_down)

            return x_down + self.epsilon * h
        else:
            return x + self.epsilon * h


class ResidualStitchv2(nn.Module):
    hidden_features: int
    output_features: int
    norm: str = 'BatchNorm'
    activation: Callable = nn.relu
    kernel_init: str = 'kaiming_out'
    training: bool = True
    strides: Tuple[int] = (2, 2)
    use_bias: bool = False
    epsilon: float = 1.0

    @nn.compact
    def __call__(self, x):
        
        h = nn.Conv(self.output_features, (3, 3), use_bias=self.use_bias,
                    strides=self.strides, kernel_init=INITS[self.kernel_init])(x)
        h = self.activation(h)
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        

        h = nn.Conv(self.output_features, (3, 3), use_bias=self.use_bias,
                    kernel_init=INITS[self.kernel_init])(h)
        h = NORMS[self.norm](use_running_average=not self.training)(h)

        if self.strides[0] != 1:
            x_down = nn.Conv(self.output_features, (1, 1), use_bias=self.use_bias,
                             strides=self.strides, kernel_init=INITS[self.kernel_init])(x)
            x_down = NORMS[self.norm](use_running_average=not self.training)(x_down)
            
            return self.activation(x_down + self.epsilon * h)
        else:
            #x = NORMS[self.norm](use_running_average=not self.training)(x)
            return self.activation(x + self.epsilon * h)