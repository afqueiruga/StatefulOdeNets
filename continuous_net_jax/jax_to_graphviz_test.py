import unittest

import jax

from .jax_to_graphviz import *


class MakeGraphTests(unittest.TestCase):
    def testSimple(self):
        def f(x, y):
            return x*y + y
        jxpr = jax.make_jaxpr(f)(1, 1)
        graph = make_graph(jxpr, name='foo')
        self.assertTrue(graph.directed)
        self.assertEqual(graph.name, 'foo')

if __name__ == "__main__":
    unittest.main()
