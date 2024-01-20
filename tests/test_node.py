import unittest

from uniflow.node import Node


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node1 = Node(name="node1", value_dict={"a": 1})
        self.node2 = Node(name="node2", value_dict={"b": 2})
        self.node3 = Node(name="node3", value_dict={"c": 3})

    def test_add_next_node(self):
        self.node1.add_next_node(self.node2)
        self.assertEqual(self.node1.next_nodes, [self.node2])

    def test_add_prev_node(self):
        self.node1.add_prev_node(self.node2)
        self.assertEqual(self.node1.prev_nodes, [self.node2])

    def test_get_all(self):
        self.node1.add_next_node(self.node2)
        self.node2.add_next_node(self.node3)
        nodes = self.node1.get_all()
        self.assertEqual(len(nodes), 3)
        self.assertIn(self.node1, nodes)
        self.assertIn(self.node2, nodes)
        self.assertIn(self.node3, nodes)

    def test_flatten(self):
        self.node1.add_next_node(self.node2)
        self.node2.add_next_node(self.node3)
        flattened_nodes = self.node1.flatten()
        self.assertEqual(len(flattened_nodes), 3)
        self.assertEqual(flattened_nodes[0]["name"], "node1")
        self.assertEqual(flattened_nodes[1]["name"], "node2")
        self.assertEqual(flattened_nodes[2]["name"], "node3")
        self.assertEqual(flattened_nodes[0]["value_dict"], {"a": 1})
        self.assertEqual(flattened_nodes[1]["value_dict"], {"b": 2})
        self.assertEqual(flattened_nodes[2]["value_dict"], {"c": 3})
