import unittest
from unittest.mock import call, patch

from uniflow.node import Node
from uniflow.op.extract.load.txt_op import ExtractTxtOp


class TestExtractTxtOp(unittest.TestCase):
    def setUp(self):
        self.extract_txt_op = ExtractTxtOp(name="extract_txt_op")

    @patch(
        "uniflow.op.extract.load.txt_op.read_file", return_value="mocked file content"
    )
    def test_call_with_empty_node(self, mock_read_file):
        # arrange
        nodes = []

        # act
        output_nodes = self.extract_txt_op(nodes)

        # assert
        mock_read_file.assert_not_called()
        self.assertEqual(len(output_nodes), 0)

    @patch(
        "uniflow.op.extract.load.txt_op.read_file", return_value="mocked file content"
    )
    def test_call_with_node_without_filename(self, mock_read_file):
        # arrange
        node = Node(name="node1", value_dict={})

        # act
        with self.assertRaises(KeyError):
            self.extract_txt_op([node])

        # assert
        mock_read_file.assert_not_called()

    @patch(
        "uniflow.op.extract.load.txt_op.read_file", return_value="mocked file content"
    )
    def test_call_with_node(self, mock_read_file):
        # arrange
        node = Node(name="node1", value_dict={"filename": "mocked_file_path"})

        # act
        output_nodes = self.extract_txt_op([node])

        # assert
        mock_read_file.assert_called_once_with("mocked_file_path")
        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], "mocked file content")

    @patch(
        "uniflow.op.extract.load.txt_op.read_file", return_value="mocked file content"
    )
    def test_call_with_multiple_nodes(self, mock_read_file):
        # arrange
        node1 = Node(name="node1", value_dict={"filename": "mocked_file_path1"})
        node2 = Node(name="node2", value_dict={"filename": "mocked_file_path2"})
        nodes = [node1, node2]

        # act
        output_nodes = self.extract_txt_op(nodes)

        # assert
        mock_read_file.assert_has_calls(
            [call("mocked_file_path1"), call("mocked_file_path2")], any_order=True
        )
        self.assertEqual(len(output_nodes), 2)
        self.assertEqual(output_nodes[0].value_dict["text"], "mocked file content")
        self.assertEqual(output_nodes[1].value_dict["text"], "mocked file content")
