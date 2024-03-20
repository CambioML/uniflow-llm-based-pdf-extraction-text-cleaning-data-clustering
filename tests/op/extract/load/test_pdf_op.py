import unittest
from unittest.mock import MagicMock

from uniflow.node import Node
from uniflow.op.extract.load.pdf_op import ExtractPDFOp, ProcessPDFOp
from uniflow.op.model.abs_model import AbsModel


class TestExtractPDFOp(unittest.TestCase):
    def test_extract_pdf_op_with_valid_node(self):
        model = MagicMock(spec=AbsModel)
        model.run.return_value = {"response": ["Extracted text"]}
        extract_op = ExtractPDFOp("test", model)
        node = Node(name="node1", value_dict={})

        result = extract_op([node])

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Node)
        self.assertEqual(result[0].value_dict["text"], "Extracted text")

    def test_extract_pdf_op_with_multiple_nodes(self):
        model = MagicMock(spec=AbsModel)
        model.run.return_value = {"response": ["Extracted text"]}
        extract_op = ExtractPDFOp("test", model)
        nodes = [Node(name="node1", value_dict={}), Node(name="node2", value_dict={})]

        result = extract_op(nodes)

        self.assertEqual(len(result), 2)
        for node in result:
            self.assertEqual(node.value_dict["text"], "Extracted text")

    def test_extract_pdf_op_with_empty_nodes(self):
        model = MagicMock(spec=AbsModel)
        extract_op = ExtractPDFOp("test", model)
        result = extract_op([])

        self.assertEqual(result, [])


class TestProcessPDFOp(unittest.TestCase):
    def setUp(self):
        self.process_op = ProcessPDFOp("process_pdf")

    def test_process_pdf_op_with_valid_text(self):
        node = Node(name="node1", value_dict={"text": "Line1\n\n\nLine2"})
        result = self.process_op([node])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Node)
        self.assertEqual(result[0].value_dict["text"], "Line1\n\nLine2")

    def test_process_pdf_op_retains_single_and_double_newlines(self):
        node = Node(name="node1", value_dict={"text": "Line1\nLine2\n\nLine3"})
        result = self.process_op([node])
        self.assertEqual(result[0].value_dict["text"], "Line1\nLine2\n\nLine3")

    def test_process_pdf_op_with_multiple_nodes(self):
        nodes = [
            Node(name="node1", value_dict={"text": "Text1\n\n\nText2"}),
            Node(name="node2", value_dict={"text": "Text3\n\n\nText4"}),
        ]
        result = self.process_op(nodes)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].value_dict["text"], "Text1\n\nText2")
        self.assertEqual(result[1].value_dict["text"], "Text3\n\nText4")

    def test_process_pdf_op_with_empty_nodes(self):
        nodes = []
        result = self.process_op(nodes)
        self.assertEqual(result, [])

    def test_process_pdf_op_with_empty_text(self):
        node = Node(name="node1", value_dict={"text": ""})
        result = self.process_op([node])
        self.assertEqual(result[0].value_dict["text"], "")


if __name__ == "__main__":
    unittest.main()
