import unittest

from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp


class TestExpandOp(unittest.TestCase):
    def test_expand_batch_inputs(self):
        # Test with one input node
        input_node = Node(name="input_node", value_dict=[{"first item": 1, "second item": 2}, 
                                                         {"third item": 3, "fourth item": 4}])
        op = ExpandOp("linear_op")
        op(input_node)
        self.assertEqual(
            input_node.flatten(),
            [
                {
                    "is_end": False,
                    "name": "input_node",
                    "next_nodes": ["linear_op_1", "linear_op_2"],
                    "prev_nodes": [],
                    "value_dict": [{"first item": 1, "second item": 2}, 
                                   {"third item": 3, "fourth item": 4}],
                },
                {
                    "is_end": False,
                    "name": "linear_op_1",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [{"first item": 1}, {"third item": 3}],
                },
                {
                    "is_end": False,
                    "name": "linear_op_2",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [{"second item": 2}, {"fourth item": 4}],
                },
            ],
        )

    def test_expand_odd_inputs(self):
        # Test with one input node
        input_node = Node(name="input_node", value_dict=[{"first item": 1, "second item": 2, "third item": 3}])
        op = ExpandOp("linear_op")
        op(input_node)
        self.assertEqual(
            input_node.flatten(),
            [
                {
                    "is_end": False,
                    "name": "input_node",
                    "next_nodes": ["linear_op_1", "linear_op_2"],
                    "prev_nodes": [],
                    "value_dict": [{"first item": 1, "second item": 2, "third item": 3}],
                },
                {
                    "is_end": False,
                    "name": "linear_op_1",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [{"first item": 1}],
                },
                {
                    "is_end": False,
                    "name": "linear_op_2",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [{"second item": 2, "third item": 3}],
                },
            ],
        )

    def test_expand_empty_input(self):
        # Test with one input node
        input_node = Node(name="input_node", value_dict=[{}])
        op = ExpandOp("linear_op")
        op(input_node)
        self.assertEqual(
            input_node.flatten(),
            [
                {
                    "is_end": False,
                    "name": "input_node",
                    "next_nodes": ["linear_op_1", "linear_op_2"],
                    "prev_nodes": [],
                    "value_dict": [{}],
                },
                {
                    "is_end": False,
                    "name": "linear_op_1",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [{}],
                },
                {
                    "is_end": False,
                    "name": "linear_op_2",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [{}],
                },
            ],
        )

    def test_expand_empty_list(self):
        # Test with one input node
        input_node = Node(name="input_node", value_dict=[])
        op = ExpandOp("linear_op")
        op(input_node)
        self.assertEqual(
            input_node.flatten(),
            [
                {
                    "is_end": False,
                    "name": "input_node",
                    "next_nodes": ["linear_op_1", "linear_op_2"],
                    "prev_nodes": [],
                    "value_dict": [],
                },
                {
                    "is_end": False,
                    "name": "linear_op_1",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [],
                },
                {
                    "is_end": False,
                    "name": "linear_op_2",
                    "next_nodes": [],
                    "prev_nodes": ["input_node"],
                    "value_dict": [],
                },
            ],
        )
