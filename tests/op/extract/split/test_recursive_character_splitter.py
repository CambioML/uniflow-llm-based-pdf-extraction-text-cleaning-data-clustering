import unittest

from uniflow.node import Node
from uniflow.op.extract.split.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)


class TestRecursiveCharacterSplitter(unittest.TestCase):
    def setUp(self):
        self.default_separators = ["\n\n", "\n"]

    def test_recursive_splitter(self):
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 10}, "test_splitter")
        text = "Hello\n\nWorld."

        chunks = splitter._recursive_splitter(text, splitter.default_separators)

        self.assertEqual(chunks, ["Hello", "World."])

    def test_merge_splits(self):
        splits = ["Hello", "World"]
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 20}, "test_splitter")

        merged = splitter._merge_splits(splits, "\n")

        self.assertEqual(merged, ["Hello\nWorld"])

    def test_recursive_splitter_with_merge_chunk(self):
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 20}, "test_splitter")
        node = Node(name="node1", value_dict={"text": "Hello World"})

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["Hello\\ World"])

    def test_recursive_splitter_with_small_chunk_size(self):
        splitter = RecursiveCharacterSplitter(
            {"max_chunk_size": 1, "chunk_overlap_size": 0}, "test_splitter"
        )
        node = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})
        expected_chunks = ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"]

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], expected_chunks)

    def test_recursive_splitter_with_no_split(self):
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 10}, "test_splitter")
        node = Node(name="node1", value_dict={"text": "HelloWorld"})

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["HelloWorld"])

    def test_recursive_splitter_with_custom_separators(self):
        splitter = RecursiveCharacterSplitter(
            {"max_chunk_size": 10, "separators": "--"}, "test_splitter"
        )
        node = Node(name="node1", value_dict={"text": "Hello--World"})

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], ["Hello", "World"])

    def test_recursive_splitter_with_large_text_default_chunk(self):
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 20}, "test_splitter")
        node = Node(name="node1", value_dict={"text": "Hello\n\nWorld\n\n" * 100})

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(len(output_nodes[0].value_dict["text"]), 100)

    def test_recursive_splitter_with_large_text_large_chunk(self):
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 9999}, "test_splitter")
        node = Node(name="node1", value_dict={"text": "Hello\n\nWorld\n\n" * 100})

        output_nodes = splitter([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(len(output_nodes[0].value_dict["text"]), 1)

    def test_special_function_call_with_multiple_nodes(self):
        splitter = RecursiveCharacterSplitter({"max_chunk_size": 10}, "test_splitter")

        node0 = Node(name="node1", value_dict={"text": "Hello\n\nWorld"})
        node1 = Node(name="node1", value_dict={"text": "Hello\n\nWorld."})
        node2 = Node(name="node1", value_dict={"text": "Hello\n\nWorld\n\n" * 10})
        node3 = Node(name="node1", value_dict={"text": "Hello\n\nWorld.\n\n" * 2})
        expected_texts = [
            ["Hello", "World"],
            ["Hello", "World."],
            ["Hello", "World"] * 10,
            ["Hello", "World.", "Hello", "World."],
        ]

        output_nodes = splitter([node0, node1, node2, node3])
        output_texts = [node.value_dict["text"] for node in output_nodes]

        self.assertEqual(output_texts, expected_texts)
