import os
import unittest
from typing import Sequence
from unittest.mock import Mock

from uniflow.op.extract.load.md_op import ExtractMarkdownOp, ProcessMarkdownOp


# Assuming the real Node and Op
class Node:
    def __init__(
        self, name: str, value_dict: dict, prev_nodes: Sequence["Node"] = None
    ):
        self.name = name
        self.value_dict = value_dict
        self.prev_nodes = prev_nodes if prev_nodes is not None else []
        self.next_nodes: list[Node] = []

    def add_next_node(self, next_node: "Node"):
        if next_node not in self.next_nodes:
            self.next_nodes.append(next_node)

    def remove_next_node(self, next_node: "Node"):
        if next_node in self.next_nodes:
            self.next_nodes.remove(next_node)


class Op:
    def unique_name(self):
        return "unique_op_name"


class TestMarkdownOps(unittest.TestCase):
    def setUp(self):
        self.test_file_1 = "test_file_1.md"
        self.test_file_2 = "test_file_2.md"
        with open(self.test_file_1, "w", encoding="utf-8") as file:
            file.write("# Title\n\nSome text\n\n\nMore text after multiple newlines.")
        with open(self.test_file_2, "w", encoding="utf-8") as file:
            file.write("Just a single line.")

    def tearDown(self):
        os.remove(self.test_file_1)
        os.remove(self.test_file_2)

    def test_extract_markdown_op(self):
        extract_op = ExtractMarkdownOp(name="extract_op")
        extract_op.unique_name = Mock(return_value="extracted_md")
        nodes = [
            Node(name="node1", value_dict={"filename": self.test_file_1}),
            Node(name="node2", value_dict={"filename": self.test_file_2}),
        ]

        extracted_nodes = extract_op(nodes)

        self.assertEqual(len(extracted_nodes), 2)
        self.assertEqual(extracted_nodes[0].name, "extracted_md")
        self.assertIn("text", extracted_nodes[0].value_dict)
        with open(self.test_file_1, "r", encoding="utf-8") as file:
            content = file.read()
        self.assertEqual(extracted_nodes[0].value_dict["text"], content)
        self.assertEqual(extracted_nodes[1].value_dict["text"], "Just a single line.")

    def test_process_markdown_op(self):
        process_op = ProcessMarkdownOp(name="process_op")
        process_op.unique_name = Mock(return_value="processed_md")

        text_node1 = Node(
            name="text_node1", value_dict={"text": "Some text\n\n\n\nNew paragraph."}
        )
        text_node2 = Node(name="text_node2", value_dict={"text": "Single line."})

        processed_nodes = process_op([text_node1, text_node2])

        self.assertEqual(len(processed_nodes), 2)
        self.assertEqual(processed_nodes[0].name, "processed_md")
        self.assertIn("text", processed_nodes[0].value_dict)
        self.assertEqual(
            processed_nodes[0].value_dict["text"], "Some text\n\nNew paragraph."
        )
        self.assertEqual(processed_nodes[1].value_dict["text"], "Single line.")

    def test_extract_markdown_no_file(self):
        extract_op = ExtractMarkdownOp(name="extract_op")
        extract_op.unique_name = Mock(return_value="extracted_md_error")
        nodes = [
            Node(name="node_error", value_dict={"filename": "non_existent_file.md"})
        ]

        with self.assertRaises(FileNotFoundError):
            extract_op(nodes)

    def test_extract_markdown_op_encoding(self):
        extract_op = ExtractMarkdownOp(name="extract_op")
        extract_op.unique_name = Mock(return_value="extracted_md")
        nodes = [
            Node(
                name="node_with_encoding",
                value_dict={"filename": self.test_file_1, "encoding": "utf-8"},
            )
        ]

        extracted_nodes = extract_op(nodes)

        self.assertEqual(len(extracted_nodes), 1)
        self.assertEqual(extracted_nodes[0].name, "extracted_md")
        self.assertIn("text", extracted_nodes[0].value_dict)

        with open(self.test_file_1, "r", encoding="utf-8") as file:
            content = file.read()
        self.assertEqual(extracted_nodes[0].value_dict["text"], content)

    def test_process_markdown_op_empty_text(self):
        process_op = ProcessMarkdownOp(name="process_op")
        process_op.unique_name = Mock(return_value="processed_md_empty")

        text_node_empty = Node(name="text_node_empty", value_dict={"text": ""})

        processed_nodes = process_op([text_node_empty])

        self.assertEqual(len(processed_nodes), 1)
        self.assertEqual(processed_nodes[0].name, "processed_md_empty")
        self.assertIn("text", processed_nodes[0].value_dict)
        self.assertEqual(processed_nodes[0].value_dict["text"], "")


if __name__ == "__main__":
    unittest.main()
