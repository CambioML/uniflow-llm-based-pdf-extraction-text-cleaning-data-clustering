import unittest
from unittest.mock import patch

from uniflow.node import Node
from uniflow.op.extract.load.txt_op import ExtractTxtOp, ProcessTxtOp


class TestExtractTxtOp(unittest.TestCase):
    def setUp(self):
        self.extract_txt_op = ExtractTxtOp(name="extract_txt_op")

    @patch(
        "uniflow.op.extract.load.txt_op.read_file", return_value="mocked file content"
    )
    def test_call(self, mock_read_file):
        # arrange
        node = Node(name="node1", value_dict={"filename": "mocked_file_path"})

        # act
        output_nodes = self.extract_txt_op([node])

        # assert
        mock_read_file.assert_called_once_with("mocked_file_path")
        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], "mocked file content")


class TestProcessTxtOp(unittest.TestCase):
    def setUp(self):
        self.process_txt_op = ProcessTxtOp(name="process_txt_op")

    def test_empty_input(self):
        node = Node(name="node1", value_dict={"text": ""})

        output_nodes = self.process_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"][0], "")

    def test_whitespace_input(self):
        node = Node(name="node1", value_dict={"text": " \n \n "})

        output_nodes = self.process_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"][0], "")

    def test_call(self):
        node = Node(
            name="node1", value_dict={"text": "This is a test\nThis is another test"}
        )

        output_nodes = self.process_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(
            output_nodes[0].value_dict["text"],
            ["This is a test", "This is another test"],
        )
