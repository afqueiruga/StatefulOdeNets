import flax
import flax.linen as nn
import jax

from .residual_modules import ResidualUnit, ResidualStitch


class ResNet(nn.Module):
    alpha: int = 8
    n_steps: int = 1
    n_classes: int = 10

    @nn.compact
    def __call__(self, x):
        alpha = self.alpha
        n_steps = self.n_steps
        h = nn.Conv(features=alpha, kernel_size=(3, 3))(x)
        for i in range(n_steps):
            h += ResidualUnit(hidden_features=alpha, output_features=alpha)(h)
        h = ResidualStitch(hidden_features=alpha,
                           output_features=2 * alpha,
                           strides=(2, 2))(h)
        for i in range(n_steps):
            h += ResidualUnit(hidden_features=2 * alpha,
                              output_features=2 * alpha)(h)
        h = ResidualStitch(hidden_features=2 * alpha,
                           output_features=4 * alpha,
                           strides=(2, 2))(h)
        for i in range(n_steps):
            h += ResidualUnit(hidden_features=4 * alpha,
                              output_features=4 * alpha)(h)
        h = nn.pooling.avg_pool(h, (4, 4), strides=(4, 4))
        h = h.reshape(h.shape[0], -1)
        h = nn.Dense(features=self.n_classes)(h)
        return nn.log_softmax(h)  # no softmax
