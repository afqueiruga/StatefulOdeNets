import flax
import jax


class ShallowNet(flax.nn.Module):

    def apply(self, x, h_dim=5, out_dim=1):
        """A shallow fully connected tanh network.

        Args:
            x: input signal.
            h_dim: the number of hidden dimensions.
            out_dim: the output dimension.

        Returns:
            The network output.
        """
        h = flax.nn.tanh(flax.nn.Dense(x, features=h_dim, bias=True))
        y = flax.nn.tanh(flax.nn.Dense(h, features=out_dim, bias=True))
        return y
