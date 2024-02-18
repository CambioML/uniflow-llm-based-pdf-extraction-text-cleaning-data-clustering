"""Test cases for ReduceOp."""

import unittest

from uniflow.node import Node
from uniflow.op.basic.reduce_op import ReduceOp


class TestReduceOp(unittest.TestCase):
    def setUp(self):
        self.reduce_fn = lambda x, y: {"value": x["value"] + y["value"]}
        self.reduce_op = ReduceOp("test_reduce", self.reduce_fn)

    def test_init(self):
        self.assertEqual(self.reduce_op._fn, self.reduce_fn)

    def test_call(self):
        node1 = Node("node1", {"value": 1})
        node2 = Node("node2", {"value": 2})

        output_nodes = self.reduce_op([(node1, node2)])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict, {"value": 3})
        self.assertEqual(output_nodes[0].prev_nodes, [node1, node2])


if __name__ == "__main__":
    unittest.main()
