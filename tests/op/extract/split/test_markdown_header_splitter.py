import unittest

from uniflow.node import Node
from uniflow.op.extract.split.markdown_header_splitter import MarkdownHeaderSplitter


class TestMarkdownHeaderSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = MarkdownHeaderSplitter("test_splitter")

    def test_special_function_call(self):
        node0 = Node(name="node1", value_dict={"text": "# Header ## Content"})
        node1 = Node(name="node1", value_dict={"text": "# Header\n## Content"})

        output_nodes = self.splitter([node0, node1])

        self.assertEqual(len(output_nodes), 2)
        self.assertEqual(output_nodes[0].value_dict["text"], ["# Header ## Content"])
        self.assertEqual(output_nodes[1].value_dict["text"], ["# Header", "## Content"])

    def test_header_splitter_basic(self):
        markdown_str = "# Header\n## Content"

        result = self.splitter.header_splitter(markdown_str)

        self.assertEqual(result, ["# Header", "## Content"])

    def test_header_splitter_multilevel_header(self):
        markdown_str = "# Header\n## Content\n# Header 2 ## Content 2"

        result = self.splitter.header_splitter(markdown_str)

        self.assertEqual(result, ["# Header", "## Content", "# Header 2 ## Content 2"])

    def test_header_splitter_with_empty_custom_headers(self):
        markdown_str = "# Header \n Content"
        custom_header = []

        result = self.splitter.header_splitter(markdown_str, custom_header)

        self.assertEqual(result, [])

    def test_header_splitter_with_invalid_custom_headers(self):
        markdown_str = "# Header</h1> \n Content"
        custom_header = [("\n", "h2")]

        result = self.splitter.header_splitter(markdown_str, custom_header)

        self.assertEqual(result, [])

    def test_header_splitter_with_no_headers(self):
        markdown_str = "\nContent with no headers"

        result = self.splitter.header_splitter(markdown_str)

        self.assertEqual(result, ["Content with no headers"])  # No split should occur
