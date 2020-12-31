import unittest

import jax
import jax.numpy as onp

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
