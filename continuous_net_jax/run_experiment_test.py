import tempfile
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

from continuous_net_jax.run_experiment import *


class RunExperimentTests(unittest.TestCase):

    def setUp(self):
        self.train_data = [
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
        ]
        self.test_data = [
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
            [onp.ones((5, 14, 14, 3)), onp.ones(5,)],
        ]

    def testContinuousNoState(self):
        with tempfile.TemporaryDirectory() as tmp:
            acc = run_an_experiment(train_data=self.train_data,
                                    validation_data=self.test_data,
                                    test_data = self.test_data,
                                    save_dir=tmp,
                                    alpha=1,
                                    hidden=1,
                                    n_step=1,
                                    n_basis=1,
                                    n_epoch=1,
                                    norm='None',
                                    learning_rate=0.1)
            self.assertEqual(acc, 1.0)

    def testContinuousStateful(self):
        print("Test testContinuousStateful")
        with tempfile.TemporaryDirectory() as tmp:
            acc = run_an_experiment(train_data=self.train_data,
                                    validation_data=self.test_data,
                                    test_data = self.test_data,
                                    save_dir=tmp,
                                    which_model='Continuous',
                                    alpha=1,
                                    hidden=1,
                                    n_step=1,
                                    n_basis=1,
                                    n_epoch=2,
                                    norm='BatchNorm')
            self.assertEqual(acc, 1.0)

    def testRefineContinuousStateful(self):
        print("Test testRefineContinuousStateful")
        with tempfile.TemporaryDirectory() as tmp:
            acc = run_an_experiment(train_data=self.train_data,
                                    validation_data=self.test_data,
                                    test_data = self.test_data,
                                    save_dir=tmp,
                                    which_model='Continuous',
                                    alpha=1,
                                    hidden=1,
                                    n_step=1,
                                    n_basis=1,
                                    n_epoch=3,
                                    norm='BatchNorm',
                                    refine_epochs=[2, 3])
            self.assertEqual(acc, 1.0)

    def testContinuousStatefulFemLinear(self):
        with tempfile.TemporaryDirectory() as tmp:
            acc = run_an_experiment(train_data=self.train_data,
                                    validation_data=self.test_data,
                                    test_data = self.test_data,
                                    save_dir=tmp,
                                    which_model='Continuous',
                                    alpha=1,
                                    hidden=1,
                                    n_step=1,
                                    scheme='Midpoint',
                                    n_basis=2,
                                    basis='fem_linear',
                                    n_epoch=1,
                                    norm='BatchNorm')
            self.assertEqual(acc, 1.0)

    def testContinuousStatefulPolyLinear(self):
        with tempfile.TemporaryDirectory() as tmp:
            acc = run_an_experiment(train_data=self.train_data,
                                    validation_data=self.test_data,
                                    test_data = self.test_data,
                                    save_dir=tmp,
                                    which_model='Continuous',
                                    alpha=1,
                                    hidden=1,
                                    n_step=1,
                                    scheme='Midpoint',
                                    n_basis=2,
                                    basis='poly_linear',
                                    n_epoch=1,
                                    norm='BatchNorm')
            self.assertEqual(acc, 1.0)

    def testResNetStateful(self):
        with tempfile.TemporaryDirectory() as tmp:
            acc = run_an_experiment(train_data=self.train_data,
                                    validation_data=self.test_data,
                                    test_data = self.test_data,
                                    save_dir=tmp,
                                    which_model='ResNet',
                                    alpha=1,
                                    hidden=1,
                                    n_step=1,
                                    n_basis=1,
                                    n_epoch=1,
                                    norm='BatchNorm')
            self.assertEqual(acc, 1.0)
