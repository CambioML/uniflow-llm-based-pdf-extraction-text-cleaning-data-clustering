"""Test cases for GroupOp."""

import unittest

from uniflow.node import Node
from uniflow.op.basic.group_op import GroupOp
from uniflow.op.prompt import Context


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        self.group_fn = lambda labels, summaries: {
            label: [s for l, s in zip(labels, summaries) if l == label]
            for label in set(labels)
        }
        self.group_op = GroupOp("test_group", self.group_fn)

    def test_init(self):
        self.assertEqual(self.group_op._fn, self.group_fn)

    def test_call(self):
        node_a0 = Node("node_a0", [Context(context=["Introduction"])])
        node_a1 = Node("node_a1", [Context(context=["Introduction"])])
        node_a2 = Node("node_a2", [Context(context=["Abstract"])])

        node_b0 = Node("node_b0", [Context(context=["A paper about life itself"])])
        node_b1 = Node("node_b1", [Context(context=["Life is complicated"])])
        node_b2 = Node("node_b2", [Context(context=["Happy wife, happy life"])])

        nodes_1 = [node_a0, node_a1, node_a2]
        nodes_2 = [node_b0, node_b1, node_b2]
        output_nodes = self.group_op(nodes_1, nodes_2)

        self.assertEqual(len(output_nodes), 2)
        self.assertEqual(
            output_nodes[0].value_dict, [Context(context="Happy wife, happy life")]
        )
        self.assertEqual(
            output_nodes[1].value_dict,
            [Context(context="A paper about life itself Life is complicated")],
        )


if __name__ == "__main__":
    unittest.main()
