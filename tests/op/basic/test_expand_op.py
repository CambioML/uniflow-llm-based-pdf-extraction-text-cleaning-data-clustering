"""Test cases for ExpandOp."""

import unittest

from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp


class TestExpandOp(unittest.TestCase):
    def setUp(self):
        self.expand_fn = lambda x: [{"value": v} for v in x["values"]]
        self.expand_op = ExpandOp("test_expand", self.expand_fn)

    def test_init(self):
        self.assertEqual(self.expand_op._fn, self.expand_fn)

    def test_call(self):
        node = Node("test_node", {"values": [1, 2, 3]})

        output_nodes = self.expand_op(node)

        self.assertEqual(len(output_nodes), 3)
        for i, output_node in enumerate(output_nodes):
            self.assertEqual(output_node.value_dict, {"value": i + 1})
            self.assertEqual(output_node.prev_nodes, [node])


if __name__ == "__main__":
    unittest.main()
