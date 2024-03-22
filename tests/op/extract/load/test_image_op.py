import unittest
from unittest.mock import Mock

from uniflow.node import Node
from uniflow.op.extract.load.image_op import ExtractImageOp, ProcessImageOp
from uniflow.op.model.abs_model import AbsModel


class TestExtractImageOp(unittest.TestCase):
    def test_extract_image_with_empty_sequence(self):
        model = Mock(spec=AbsModel)
        op = ExtractImageOp("test_op", model)
        self.assertEqual(op([]), [])

    def test_extract_image_with_single_node(self):
        model = Mock(spec=AbsModel)
        model.run = Mock(return_value={"response": ["Extracted text"]})

        node = Node("input_node", {"data": "image_data"})
        op = ExtractImageOp("test_op", model)
        result_nodes = op([node])

        self.assertEqual(len(result_nodes), 1)
        self.assertEqual(result_nodes[0].value_dict["text"], "Extracted text")
        self.assertIn(node, result_nodes[0].prev_nodes)

    def test_extract_image_with_multiple_nodes(self):
        model = Mock(spec=AbsModel)
        model.run = Mock(return_value={"response": ["Extracted text"]})

        nodes = [Node(f"input_node_{i}", {"data": "image_data"}) for i in range(3)]
        op = ExtractImageOp("test_op", model)
        result_nodes = op(nodes)

        self.assertEqual(len(result_nodes), 3)
        for i, result_node in enumerate(result_nodes):
            self.assertEqual(result_node.value_dict["text"], "Extracted text")
            self.assertIn(nodes[i], result_node.prev_nodes)


class TestProcessImageOp(unittest.TestCase):
    def test_process_image_with_empty_sequence(self):
        op = ProcessImageOp("test_op")
        self.assertEqual(op([]), [])

    def test_process_image_with_single_node(self):
        node = Node("input_node", {"text": "Hello\n\n\nWorld"})
        op = ProcessImageOp("test_op")
        result_nodes = op([node])

        self.assertEqual(len(result_nodes), 1)
        self.assertEqual(result_nodes[0].value_dict["text"], "Hello\n\nWorld")
        self.assertIn(node, result_nodes[0].prev_nodes)

    def test_process_image_with_multiple_nodes(self):
        nodes = [
            Node(f"input_node_{i}", {"text": f"Text with\n\n\n\n{i} newlines\n\n\n"})
            for i in range(3)
        ]
        op = ProcessImageOp("test_op")
        result_nodes = op(nodes)

        self.assertEqual(len(result_nodes), 3)
        for i, result_node in enumerate(result_nodes):
            self.assertEqual(
                result_node.value_dict["text"], f"Text with\n\n{i} newlines"
            )
            self.assertIn(nodes[i], result_node.prev_nodes)

    def test_process_image_with_leading_and_trailing_whitespace(self):
        node = Node("input_node", {"text": "\n\n\n   Hello World   \n\n\n\n"})
        op = ProcessImageOp("test_op")
        result_nodes = op([node])

        self.assertEqual(result_nodes[0].value_dict["text"], "Hello World")
        self.assertIn(node, result_nodes[0].prev_nodes)

    def test_process_image_without_extra_newlines(self):
        node = Node("input_node", {"text": "Hello\nWorld"})
        op = ProcessImageOp("test_op")
        result_nodes = op([node])

        self.assertEqual(result_nodes[0].value_dict["text"], "Hello\nWorld")
        self.assertIn(node, result_nodes[0].prev_nodes)

    def test_process_image_with_only_whitespace(self):
        node = Node("input_node", {"text": " \n\n\n  \n \n "})
        op = ProcessImageOp("test_op")
        result_nodes = op([node])

        self.assertEqual(result_nodes[0].value_dict["text"], "")
        self.assertIn(node, result_nodes[0].prev_nodes)


if __name__ == "__main__":
    unittest.main()
