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

    def test_recursive_splitter_with_small_chunk_size(self):
        splitter = RecursiveCharacterSplitter("test_splitter", chunk_size=1)
        text = "Hello\n\nWorld"
        expected_chunks = ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"]

        chunks = splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(chunks, expected_chunks)

    def test_recursive_splitter_with_zero_chunk_size(self):
        splitter = RecursiveCharacterSplitter("test_splitter", chunk_size=0)
        text = "Hello\n\nWorld"
        expected_chunks = ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"]

        chunks = splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(chunks, expected_chunks)

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

    def test_recursive_splitter_with_large_text_default_chunk(self):
        text = "Hello\n\nWorld\n\n" * 100

        chunks = self.splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(len(chunks), 100)

    def test_recursive_splitter_with_large_text_large_chunk(self):
        splitter = RecursiveCharacterSplitter("test_splitter", chunk_size=9999)
        text = "Hello\n\nWorld\n\n" * 100

        chunks = splitter._recursive_splitter(text, self.default_separators)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks, ["HelloWorld" * 100])

    def test_special_function_call(self):
        node = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})
        output_nodes = self.splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["HelloWorld"])

    def test_special_function_call_with_multiple_nodes(self):
        node0 = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})
        node1 = Node(name="node1", value_dict={"text": "Hello\n\nWorld."})
        node2 = Node(name="node1", value_dict={"text": "Hello\n\nWorld\n\n" * 10})
        node3 = Node(name="node1", value_dict={"text": "Hello\n\nWorld.\n\n" * 2})
        expected_texts = [
            ["HelloWorld"],
            ["Hello", "World."],
            ["HelloWorld"] * 10,
            ["Hello", "World.", "Hello", "World."],
        ]

        output_nodes = self.splitter([node0, node1, node2, node3])
        output_texts = [node.value_dict["text"] for node in output_nodes]

        self.assertEqual(output_texts, expected_texts)
