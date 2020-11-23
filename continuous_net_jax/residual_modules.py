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
