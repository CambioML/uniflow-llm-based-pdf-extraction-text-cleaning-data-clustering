import unittest
from unittest.mock import patch

from uniflow.node import Node
from uniflow.op.extract.load.html_op import ExtractHTMLOp, ProcessHTMLOp


class TestExtractHTMLOp(unittest.TestCase):
    def test_extract_html_op_with_url(self):
        extract_op = ExtractHTMLOp("test_extract")
        node = Node("test_node", {"url": "http://testsite.com"})
        with patch(
            "uniflow.op.extract.load.html_op.read_file",
            return_value="<html><body><p>Hello World</p></body></html>",
        ):
            output = extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello World")

    def test_extract_html_op_with_filename(self):
        extract_op = ExtractHTMLOp("test_extract")
        node = Node("test_node", {"filename": "testfile.html"})
        with patch(
            "uniflow.op.extract.load.html_op.read_file",
            return_value="<html><body><p>Hello File</p></body></html>",
        ):
            output = extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello File")

    def test_extract_html_op_with_no_url_or_filename(self):
        extract_op = ExtractHTMLOp("test_extract")
        node = Node("test_node", {})
        with self.assertRaises(ValueError):
            extract_op([node])

    def test_extract_html_op_with_container(self):
        extract_op = ExtractHTMLOp("test_extract")
        node = Node("test_node", {"url": "http://testsite.com"})
        html_content = "<html><body><div>Hello, <span>World!</span></div></body></html>"
        with patch(
            "uniflow.op.extract.load.html_op.read_file", return_value=html_content
        ):
            output = extract_op([node])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].value_dict["text"], "Hello, World!")

    def test_extract_html_op_with_empty_container(self):
        extract_op = ExtractHTMLOp("test_extract")
        node = Node("test_node", {"url": "http://testsite.com"})

        html_content = "<html><body><div><span></span></div></body></html>"
        with patch(
            "uniflow.op.extract.load.html_op.read_file", return_value=html_content
        ):
            output = extract_op([node])
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
