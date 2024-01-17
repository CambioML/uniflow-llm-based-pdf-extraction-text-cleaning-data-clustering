import unittest
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class DummyOp(Op):
    def __call__(self, nodes: Sequence[Node]):
        return nodes


class TestOp(unittest.TestCase):
    def setUp(self):
        self.op_name = "test_op"
        self.op = DummyOp(self.op_name)

    # def tearDown(self):
    #     utils.OPS_NAME.clear()

    def test_init(self):
        self.assertEqual(self.op._scope_name, self.op_name)
        self.assertEqual(self.op._count, 0)

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
        self.assertEqual(self.op.unique_name(), f"{self.op_name}_1")
        self.assertEqual(self.op.unique_name(), f"{self.op_name}_2")

    def test_call(self):
        node = Node(name="node1", value_dict={"a": 1})
        return_node = self.op([node])
        self.assertEqual(node, return_node[0])
