import unittest

import flax.linen as nn
import jax
import jax.numpy as onp
from jax.config import config
config.enable_omnistaging()

from .tools import *


class ToolsTest(unittest.TestCase):

    def test_count_parameters(self):
        # One number in various trees.
        self.assertEqual(1, count_parameters(2.0))
        self.assertEqual(1, count_parameters([2.0]))
        self.assertEqual(1, count_parameters({'a': {'b': 2.0}}))

        # One array in various trees.
        self.assertEqual(15, count_parameters(onp.zeros((3, 5))))
        self.assertEqual(15, count_parameters([onp.zeros((3, 5))]))
        self.assertEqual(15, count_parameters({'a': {'b': onp.zeros((3, 5))}}))

        # Multiple arrays in different trees.
        self.assertEqual(16, count_parameters([2.0, onp.zeros((3, 5))]))
        self.assertEqual(
            15 + 77,
            count_parameters({
                'a': {
                    'b': onp.zeros((3, 5))
                },
                'c': onp.zeros((7, 11)),
            }))

    def test_module_to_dict(self):
        F = nn.Dense(10)
        module_to_dict(F)
