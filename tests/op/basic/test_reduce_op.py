import unittest

from uniflow.node import Node
from uniflow.op.basic.reduce_op import ReduceOp


class TestReduceOp(unittest.TestCase):
    def test_call(self):
        node_1 = Node(name="Expand_1", value_dict={"x": 1})
        node_2 = Node(name="Expand_2", value_dict={"y": 2})
        op = ReduceOp(name="reduce_op")
        op(node_1, node_2)
        self.assertEqual(
            node_1.flatten(),
            [
                {
                    "is_end": False,
                    "name": "Expand_1",
                    "next_nodes": ["Reduce_1"],
                    "prev_nodes": [],
                    "value_dict": {"x": 1},
                },
                {
                    "is_end": False,
                    "name": "Reduce_1",
                    "next_nodes": [],
                    "prev_nodes": ["Expand_1", "Expand_2"],
                    "value_dict": {"x y": "1 2"},
                },
            ],
        )
        self.assertEqual(
            node_2.flatten(),
            [
                {
                    "is_end": False,
                    "name": "Expand_2",
                    "next_nodes": ["Reduce_1"],
                    "prev_nodes": [],
                    "value_dict": {"y": 2},
                },
                {
                    "is_end": False,
                    "name": "Reduce_1",
                    "next_nodes": [],
                    "prev_nodes": ["Expand_1", "Expand_2"],
                    "value_dict": {"x y": "1 2"},
                },
            ],
        )
