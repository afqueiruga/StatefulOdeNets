# This is the unittest for the convergence test; convergence.py is the
# convergence "test".

import unittest

from jax.config import config
config.enable_omnistaging()

from .convergence import *

def ConvergenceTestTests(unittest.TestCase):

    def testConvergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            pass
