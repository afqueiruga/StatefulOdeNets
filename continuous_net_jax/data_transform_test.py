import unittest
import jax.numpy as jnp

from data_transform import *


class DataTransformTest(unittest.TestCase):

    def test_data(self):
        data = [[jnp.ones((3, 1, 28, 28)), jnp.ones(3)]]
        for X, Y in DataTransform(data):
            self.assertEqual(X.shape, (3, 28, 28, 1))

    def test_data_channels(self):
        data = [
            [jnp.ones((13, 8, 32, 32)), jnp.ones(3)],
            [jnp.ones((13, 8, 32, 32)), jnp.ones(3)],
        ]
        for X, Y in DataTransform(data):
            self.assertEqual(X.shape, (13, 32, 32, 8))

    def test_multi_epoch(self):
        data = [
            [jnp.ones((13, 8, 32, 32)), jnp.ones(3)],
            [jnp.ones((13, 8, 32, 32)), jnp.ones(3)],
        ]
        data_transform = DataTransform(data)
        for X, Y in data_transform:
            self.assertEqual(X.shape, (13, 32, 32, 8))
        for X, Y in data_transform:
            self.assertEqual(X.shape, (13, 32, 32, 8))


if __name__ == "__main__":
    unittest.main()
