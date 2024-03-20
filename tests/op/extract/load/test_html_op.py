import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from uniflow.node import Node
from uniflow.op.extract.load.html_op import ExtractHTMLOp, ProcessHTMLOp


class TestExtractHTMLOp(unittest.TestCase):
    def setUp(self):
        self.extract_op = ExtractHTMLOp("test_extract")

    def test_bs4(self):
        soup = BeautifulSoup("<html></html>", "html.parser")
        return soup

    def test_extract_html_op_with_url(self):
        node = Node("test_node", {"url": "http://testsite.com"})
        with patch(
            "uniflow.op.extract.load.html_op.read_file",
            return_value="<html><body><p>Hello World</p></body></html>",
        ):
            output = self.extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello World")

    @patch("uniflow.op.extract.load.html_op.read_file")
    def test_call_with_multiple_nodes(self, mock_read_file):
        # arrange
        nodes = [
            Node("test_node_1", {"url": "http://testsite1.com"}),
            Node("test_node_2", {"url": "http://testsite2.com"}),
        ]

        mock_read_file.side_effect = [
            "<html><body><p>Hello Site 1</p></body></html>",
            "<html><body><p>Hello Site 2</p></body></html>",
        ]

        # act
        output_nodes = self.extract_op(nodes)

        # assert
        self.assertEqual(mock_read_file.call_count, 2)
        self.assertEqual(len(output_nodes), 2)
        self.assertEqual(output_nodes[0].value_dict["text"], "Hello Site 1")
        self.assertEqual(output_nodes[1].value_dict["text"], "Hello Site 2")

    @patch("uniflow.op.extract.load.html_op.read_file")
    def test_call_with_empty_node(self, mock_read_file):
        # arrange
        nodes = []

        # act
        output_nodes = self.extract_op(nodes)

        # assert
        mock_read_file.assert_not_called()
        self.assertEqual(len(output_nodes), 0)

    def test_extract_html_op_with_filename(self):
        node = Node("test_node", {"filename": "testfile.html"})
        with patch(
            "uniflow.op.extract.load.html_op.read_file",
            return_value="<html><body><p>Hello File</p></body></html>",
        ):
            output = self.extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello File")

    def test_extract_html_op_with_no_url_or_filename(self):
        node = Node("test_node", {})
        with self.assertRaises(ValueError):
            self.extract_op([node])

    def test_extract_html_op_with_container(self):
        node = Node("test_node", {"url": "http://testsite.com"})
        html_content = "<html><body><div>Hello, <span>World!</span></div></body></html>"
        with patch(
            "uniflow.op.extract.load.html_op.read_file", return_value=html_content
        ):
            output = self.extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello, World!")

    def test_extract_html_op_with_empty_container(self):
        node = Node("test_node", {"url": "http://testsite.com"})

        html_content = "<html><body><div><span></span></div></body></html>"
        with patch(
            "uniflow.op.extract.load.html_op.read_file", return_value=html_content
        ):
            output = self.extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "")


class TestProcessHTMLOp(unittest.TestCase):
    def test_process_html_op(self):
        process_op = ProcessHTMLOp("test_process")
        node = Node("test_node", {"text": "\n Hello World \n"})
        output = process_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello World")


if __name__ == "__main__":
    unittest.main()
