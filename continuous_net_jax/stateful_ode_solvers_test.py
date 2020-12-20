import unittest

from .stateful_ode_solvers import *

def stateful_f(p, x):
    return p + 1.0, x + p + 1.0

class StatefulIntegratorsTests(unittest.TestCase):

    def test_euler(self):
        params_of_t = lambda t: 2.0
        state, x2 = Euler(params_of_t, 3.0, 0.0, stateful_f, 1.0)
        self.assertEqual(state, 3.0)
        self.assertEqual(x2, 9.0)

if __name__ == "__main__":
    unittest.main()
