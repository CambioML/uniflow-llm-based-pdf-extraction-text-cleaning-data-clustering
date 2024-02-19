import unittest
from uniflow.node import Node
from uniflow.op.extract.split.pattern_splitter_op import PatternSplitter


class TestPatternSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = PatternSplitter("test_splitter")

    def test_call(self):
        node = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})

        output_nodes = self.splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["Hello", "World"])

    def test_call_with_custom_splitter(self):
        splitter = PatternSplitter("test_splitter", splitter=" ")
        node = Node(name="node1", value_dict={"text": "Hello World"})

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["Hello", "World"])

    def test_call_with_no_split(self):
        node = Node(name="node1", value_dict={"text": "HelloWorld"})

        output_nodes = self.splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["HelloWorld"])
