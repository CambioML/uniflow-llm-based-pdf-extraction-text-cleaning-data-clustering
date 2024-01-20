import unittest

from uniflow.node import Node
from uniflow.viz import Viz


class TestNode(unittest.TestCase):
    def setUp(self):
        pass

    def test_to_digraph(self):
        # Create nodes
        node1 = Node(name="node1", value_dict={"a": 1})
        node2 = Node(name="node2", value_dict={"b": 2})
        node3 = Node(name="node3", value_dict={"c": 3})
        node4 = Node(name="node4", value_dict={"d": 4})

        # Connect nodes
        node1.add_next_node(node2)
        node2.add_next_node(node3)
        node2.add_next_node(node4)

        # Test to_digraph method
        graph = Viz.to_digraph(node1)
        print(str(graph))
        expected_output = "digraph {\n\tnode1\n\tnode1 -> node2\n\tnode2\n\tnode2 -> node3\n\tnode2 -> node4\n\tnode3\n\tnode4\n}\n"
        self.assertEqual(str(graph), expected_output)
