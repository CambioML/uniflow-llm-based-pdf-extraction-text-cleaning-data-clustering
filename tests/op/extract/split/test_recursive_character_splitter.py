import unittest

from uniflow.node import Node
from uniflow.op.extract.split.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)


class TestRecursiveCharacterSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = RecursiveCharacterSplitter("test_splitter", chunk_size=10)
        self.default_separators = ["\n\n", "\n", " ", ""]

    def test_recursive_splitter(self):
        text = "Hello\n\nWorld."

        chunks = self.splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(chunks, ["Hello", "World."])

    def test_recursive_splitter_with_merge_chunk(self):
        splitter = RecursiveCharacterSplitter("test_splitter", chunk_size=100)
        text = "Hello\n\nWorld"

        chunks = splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(chunks, ["HelloWorld"])

    def test_recursive_splitter_with_small_chunk(self):
        splitter0 = RecursiveCharacterSplitter("test_splitter", chunk_size=0)
        splitter1 = RecursiveCharacterSplitter("test_splitter", chunk_size=1)
        text = "Hello\n\nWorld"
        expected_chunks = ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"]

        chunks0 = splitter0._recursive_splitter(text, self.default_separators)
        chunks1 = splitter1._recursive_splitter(text, self.default_separators)

        self.assertEqual(chunks0, expected_chunks)
        self.assertEqual(chunks1, expected_chunks)

    def test_recursive_splitter_with_no_separators(self):
        text = "Hello\n\nWorld"
        separators = []

        chunks = self.splitter._recursive_splitter(text, separators)

        self.assertEqual(chunks, [])

    def test_recursive_splitter_with_no_split(self):
        text = "HelloWorld"

        chunks = self.splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(chunks, ["HelloWorld"])

    def test_recursive_splitter_with_custom_separators(self):
        text = "Hello--World."
        separators = ["-", " "]

        chunks = self.splitter._recursive_splitter(text, separators)

        self.assertEqual(chunks, ["Hello", "World."])

    def test_recursive_splitter_with_large_text(self):
        splitter0 = RecursiveCharacterSplitter("test_splitter", chunk_size=1)
        splitter1 = RecursiveCharacterSplitter("test_splitter", chunk_size=9999)
        text = "Hello\n\nWorld\n\n" * 100

        chunks = self.splitter._recursive_splitter(text, self.default_separators)
        chunks0 = splitter0._recursive_splitter(text, self.default_separators)
        chunks1 = splitter1._recursive_splitter(text, self.default_separators)

        self.assertEqual(len(chunks), 100)
        self.assertEqual(len(chunks0), 10 * 100)
        self.assertEqual(len(chunks1), 1)
        self.assertEqual(chunks1, ['HelloWorld' * 100])

    def test_call(self):
        node = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})
        output_nodes = self.splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["HelloWorld"])

    def test_call_with_multiple_nodes(self):
        node0 = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})
        expected0 = ["HelloWorld"]
        node1 = Node(name="node1", value_dict={"text": "Hello\n\nWorld."})
        expected1 = ["Hello", "World."]
        node2 = Node(name="node1", value_dict={"text": "Hello\n\nWorld\n\n" * 10})
        expected2 = ["HelloWorld"] * 10
        node3 = Node(name="node1", value_dict={"text": "Hello\n\nWorld.\n\n" * 2})
        expected3 = ["Hello", "World.", "Hello", "World."]

        output_nodes = self.splitter([node0, node1, node2, node3])

        self.assertEqual(len(output_nodes), 4)
        self.assertEqual(output_nodes[0].value_dict["text"], expected0)
        self.assertEqual(output_nodes[1].value_dict["text"], expected1)
        self.assertEqual(output_nodes[2].value_dict["text"], expected2)
        self.assertEqual(output_nodes[3].value_dict["text"], expected3)
