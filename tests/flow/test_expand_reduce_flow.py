import unittest

from uniflow.flow.expand_reduce_flow import ExpandReduceFlow
from uniflow.node import Node


class TestExpandReduceFlow(unittest.TestCase):
    def test_expand_reduce_flow(self):
        node = Node(name="root", value_dict={"x": 1, "y": 2})
        flow = ExpandReduceFlow()
        flow.run(node)
        self.assertEqual(
            node.flatten(),
            [
                {
                    "name": "root",
                    "next_nodes": ["Expand_1", "Expand_2"],
                    "prev_nodes": [],
                    "value_dict": {"x": 1, "y": 2},
                    "is_end": False,
                },
                {
                    "name": "Expand_1",
                    "next_nodes": [],
                    "prev_nodes": ["root"],
                    "value_dict": {"x": 1},
                    "is_end": False,
                },
                {
                    "name": "Expand_2",
                    "next_nodes": [],
                    "prev_nodes": ["root"],
                    "value_dict": {"y": 2},
                    "is_end": False,
                },
                {
                    "name": "Reduce_1",
                    "next_nodes": [],
                    "prev_nodes": ["Expand_1", "Expand_2"],
                    "value_dict": {"x y": "1 2"},
                    "is_end": False,
                },
            ],
        )
