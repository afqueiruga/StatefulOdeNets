import unittest
from odenet import *

class TestODE(unittest.TestCase):
    def test_LinearODE(self):
        lin = LinearODE(2,3,4)
        self.assertEqual(lin.weights.shape[0],2)
        self.assertEqual(lin.weights.shape[1],3)
        self.assertEqual(lin.weights.shape[2],4)
        lin.weights[:]=2.0
        lin.bias[:]=2.0
        lin2 = lin.refine()
        self.assertEqual(lin2.weights.shape[0],4)
        self.assertEqual(lin2.weights.shape[1],3)
        self.assertEqual(lin2.weights.shape[2],4)
        self.assertEqual(lin2.weights[0,0,0], lin.weights[0,0,0])
        self.assertEqual(lin2.weights[1,0,0], lin.weights[0,0,0])
        
if __name__=='__main__':
    unittest.main()