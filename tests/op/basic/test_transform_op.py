"""Test cases for TransformOp."""

import unittest
from unittest.mock import Mock

from uniflow.node import Node
from uniflow.op.basic.transform_op import TransformOp


class TestTransformOp(unittest.TestCase):
    def setUp(self):
        self.transform_fn = Mock(return_value={"transformed": True})
        self.transform_op = TransformOp("test_op", self.transform_fn)

    def test_init(self):
        self.assertEqual(self.transform_op._fn, self.transform_fn)

    def test_call(self):
        mock_node = Mock(spec=Node)
        mock_node.value_dict = {"original": True}

        output_nodes = self.transform_op([mock_node])

        self.transform_fn.assert_called_once_with(mock_node.value_dict)
        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict, {"transformed": True})
        self.assertEqual(output_nodes[0].prev_nodes, [mock_node])


if __name__ == "__main__":
    unittest.main()
