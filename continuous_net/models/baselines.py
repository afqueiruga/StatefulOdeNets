import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ..residual_modules import ResidualUnit, ResidualStitch, NORMS, INITS


class ResNet(nn.Module):
    alpha: int = 8
    hidden: int = 8  # Not using this variable.
    n_classes: int = 10
    n_step: int = 2
    norm: str = "BatchNorm"
    kernel_init: str = 'kaiming_out'
    training: bool = True

    @nn.compact
    def __call__(self, x):
        alpha = self.alpha
        n_step = self.n_step
        h = nn.Conv(features=alpha,
                    kernel_size=(3, 3),
                    use_bias=False,
                    kernel_init=INITS[self.kernel_init])(x)
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        h = nn.relu(h)
        for i in range(n_step):
            h += ResidualUnit(hidden_features=alpha,
                              norm=self.norm,
                              kernel_init=self.kernel_init,
                              training=self.training)(h)
        h = ResidualStitch(hidden_features=alpha,
                           output_features=2 * alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           kernel_init=self.kernel_init,
                           training=self.training)(h)
        for i in range(n_step):
            h += ResidualUnit(hidden_features=2 * alpha,
                              norm=self.norm,
                              kernel_init=self.kernel_init,
                              training=self.training)(h)
        h = ResidualStitch(hidden_features=2 * alpha,
                           output_features=4 * alpha,
                           strides=(2, 2),
                           norm=self.norm,
                           kernel_init=self.kernel_init,
                           training=self.training)(h)
        for i in range(n_step):
            h += ResidualUnit(hidden_features=4 * alpha,
                              norm=self.norm,
                              kernel_init=self.kernel_init,
                              training=self.training)(h)
        h = NORMS[self.norm](use_running_average=not self.training)(h)
        # h = nn.pooling.avg_pool(h, (h.shape[-3], h.shape[-2]))
        # h = h.reshape(h.shape[0], -1)
        h = jnp.mean(h, axis=(1, 2))
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax
