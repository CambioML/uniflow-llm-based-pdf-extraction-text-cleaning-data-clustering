import unittest

from uniflow.node.node import Node
from uniflow.op.basic.linear_op import LinearOp


class TestLinearOp(unittest.TestCase):
    def test_call(self):
        # Test with one input node
        input_node = Node(name="input_node", value_dict={"x": 1})
        op = LinearOp("linear_op")
        op([input_node])
        op([input_node])
        self.assertEqual(
            input_node.flatten(),
            [
                {
                    "is_end": False,
                    "name": "input_node",
                    "next_nodes": ["linear_op_1", "linear_op_2"],
                    "prev_nodes": [],
                    "value_dict": {"x": 1},
                },
                {
                    "is_end": False,
                    "name": "linear_op_1",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": {"x": 1},
                },
                {
                    "is_end": False,
                    "name": "linear_op_2",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": {"x": 1},
                },
            ],
        )
