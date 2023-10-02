from typing import Sequence
import unittest
from unittest.mock import MagicMock, patch

from uniflow.op.op import Op
import uniflow.op.utils as utils
from uniflow.node.node import Node


class DummyOp(Op):
    def __call__(self, nodes: Sequence[Node]):
        return nodes


class TestOp(unittest.TestCase):
    def setUp(self):
        self.op_name = "test_op"
        self.op = DummyOp(self.op_name)

    def test_init(self):
        self.assertEqual(self.op._scope_name, self.op_name)
        self.assertEqual(self.op._count, 0)
        self.assertIn(self.op_name, utils.OPS_NAME)

        with self.assertRaises(ValueError):
            DummyOp(self.op_name)

    def test_del(self):
        self.assertIn(self.op_name, utils.OPS_NAME)
        del self.op
        self.assertNotIn(self.op_name, utils.OPS_NAME)

    def test_clear_count(self):
        self.op._count = 10
        self.op.clear_count()
        self.assertEqual(self.op._count, 0)

    def test_increase_count(self):
        self.op._count = 10
        self.op._increase_count()
        self.assertEqual(self.op._count, 11)

    def test_name(self):
        self.assertEqual(self.op.name(), self.op_name)

    def test_unique_name(self):
        self.assertEqual(self.op.unique_name(), f"{self.op_name}:1")
        self.assertEqual(self.op.unique_name(), f"{self.op_name}:2")

    def test_call(self):
        node = Node(name="node1", value_dict={"a": 1})
        return_node = self.op([node])
        self.assertEqual(node, return_node[0])
