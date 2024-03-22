import unittest
from unittest.mock import MagicMock, mock_open, patch

from uniflow.node import Node
from uniflow.op.extract.load.ipynb_op import ExtractIpynbOp, ProcessIpynbOp


class TestExtractIpynbOp(unittest.TestCase):
    @patch("nbformat.read")
    @patch("nbconvert.MarkdownExporter")
    def test_extract_with_valid_nodes(self, mock_markdown_exporter, mock_nbformat_read):
        mock_file_content = '{"cells": [{"cell_type": "markdown", "source": "Some markdown content"}], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}'
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            mock_nb = MagicMock()
            mock_nbformat_read.return_value = mock_nb

            mock_md_exporter_instance = mock_markdown_exporter.return_value
            mock_md_exporter_instance.from_notebook_node.return_value = (
                "# Converted Markdown",
                None,
            )

            extract_op = ExtractIpynbOp("test_op")
            test_nodes = [
                Node(name="test_node_1", value_dict={"filename": "dummy.ipynb"})
            ]

            output_nodes = extract_op(test_nodes)

            self.assertEqual(len(output_nodes), 1)
            self.assertIn("# Converted Markdown", output_nodes[0].value_dict["text"])
            mock_nbformat_read.assert_called_once()
            mock_md_exporter_instance.from_notebook_node.assert_called_once()

    def test_extract_ipynb_with_no_nodes(self):
        op = ExtractIpynbOp("extract_ipynb")
        result_nodes = op([])
        self.assertEqual(result_nodes, [])

    def test_extract_ipynb_with_missing_file(self):
        op = ExtractIpynbOp("extract_ipynb")
        mock_node = Node(
            name="test_node", value_dict={"filename": "non_existent.ipynb"}
        )
        nodes = [mock_node]

        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                op(nodes)

    def test_extract_ipynb_with_invalid_file_content(self):
        op = ExtractIpynbOp("extract_ipynb")
        mock_node = Node(
            name="test_node", value_dict={"filename": "invalid_content.ipynb"}
        )
        nodes = [mock_node]

        with patch("builtins.open", mock_open(read_data="invalid")), patch(
            "nbformat.read", side_effect=ValueError
        ):
            with self.assertRaises(ValueError):
                op(nodes)


class TestProcessIpynbOp(unittest.TestCase):
    def test_process_ipynb_with_valid_text(self):
        op = ProcessIpynbOp("process_ipynb")
        mock_node = Node(name="test_node", value_dict={"text": "\nValid text\n"})
        nodes = [mock_node]
        result_nodes = op(nodes)

        self.assertEqual(len(result_nodes), 1)
        self.assertEqual(result_nodes[0].value_dict["text"], "Valid text")

    def test_process_ipynb_with_empty_text(self):
        op = ProcessIpynbOp("process_ipynb")
        mock_node = Node(name="test_node", value_dict={"text": ""})
        nodes = [mock_node]
        result_nodes = op(nodes)

        self.assertEqual(len(result_nodes), 1)
        self.assertEqual(result_nodes[0].value_dict["text"], "")

    def test_process_ipynb_with_no_nodes(self):
        op = ProcessIpynbOp("process_ipynb")
        result_nodes = op([])
        self.assertEqual(result_nodes, [])

    def test_process_ipynb_with_whitespace_only_text(self):
        op = ProcessIpynbOp("process_ipynb")
        mock_node = Node(name="test_node", value_dict={"text": "  \n  \t "})
        nodes = [mock_node]
        result_nodes = op(nodes)

        self.assertEqual(len(result_nodes), 1)
        self.assertEqual(result_nodes[0].value_dict["text"], "")


if __name__ == "__main__":
    unittest.main()
