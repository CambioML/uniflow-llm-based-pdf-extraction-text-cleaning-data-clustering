"""Test cases for GroupOp."""

import unittest

from uniflow.node import Node
from uniflow.op.basic.group_op import GroupOp
from uniflow.op.prompt import Context


class TestGroupOp(unittest.TestCase):
    def setUp(self):
        self.preprocess_fn = lambda nodes_1, nodes_2: [
            (
                node_label.value_dict["response"][0],
                node_summary.value_dict["response"][0],
            )
            for node_label, node_summary in zip(nodes_1, nodes_2)
        ]
        self.group_fn = lambda labels, summaries: {
            label: [s for l, s in zip(labels, summaries) if l == label]
            for label in set(labels)
        }
        self.group_op = GroupOp("test_group", self.preprocess_fn, self.group_fn)

    def test_init(self):
        self.assertEqual(self.group_op._preprocess_fn, self.preprocess_fn)
        self.assertEqual(self.group_op._fn, self.group_fn)

    def test_call(self):
        node_a0 = Node("node_a0", {"response": ["Introduction"]})
        node_a1 = Node("node_a1", {"response": ["Introduction"]})
        node_a2 = Node("node_a2", {"response": ["Abstract"]})

        node_b0 = Node("node_b0", {"response": ["A paper about life itself"]})
        node_b1 = Node("node_b1", {"response": ["Life is complicated"]})
        node_b2 = Node("node_b2", {"response": ["Happy wife, happy life"]})

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
