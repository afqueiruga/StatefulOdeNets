import unittest

from .nonauto_ode_solvers import *

X0 = 1.0
T0 = 0.0
DT = 1.0

def f(_, x):
    return 1.0



class IntegratorsTest(unittest.TestCase):
    def testEuler(self):
        self.assertAlmostEqual(Euler(lambda t: 0, X0, T0, f, DT), 2.0)
    def testMidpoint(self):
        self.assertAlmostEqual(Midpoint(lambda t: 0, X0, T0, f, DT), 2.0)
    def testRK4(self):
        self.assertAlmostEqual(RK4(lambda t: 0, X0, T0, f, DT), 2.0)
    def testRK4_38(self):
        self.assertAlmostEqual(RK4_38(lambda t: 0, X0, T0, f, DT), 2.0)

if __name__ == "__main__":
    unittest.main()
