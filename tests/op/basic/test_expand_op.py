import unittest

from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp


class TestExpandOp(unittest.TestCase):
    def test_call(self):
        input_node = Node(name="root", value_dict={"x": 1, "y": 2})
        op = ExpandOp(name="expand_op")
        op(input_node)
        self.assertEqual(
            input_node.flatten(),
            [
                {
                    "is_end": False,
                    "name": "root",
                    "next_nodes": ["Expand_1", "Expand_2"],
                    "prev_nodes": [],
                    "value_dict": {"x": 1, "y": 2},
                },
                {
                    "is_end": False,
                    "name": "Expand_1",
                    "next_nodes": [],
                    "prev_nodes": ["root"],
                    "value_dict": {"x": 1},
                },
                {
                    "is_end": False,
                    "name": "Expand_2",
                    "next_nodes": [],
                    "prev_nodes": ["root"],
                    "value_dict": {"y": 2},
                },
            ],
        )
