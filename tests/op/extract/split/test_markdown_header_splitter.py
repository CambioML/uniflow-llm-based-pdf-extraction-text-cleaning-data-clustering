import unittest

from uniflow.node import Node
from uniflow.op.extract.split.markdown_header_splitter import MarkdownHeaderSplitter


class TestMarkdownHeaderSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = MarkdownHeaderSplitter("test_splitter")

    def test_call(self):
        node0 = Node(name="node1", value_dict={"text": "# Header ## Content"})
        node1 = Node(name="node1", value_dict={"text": "# Header\n## Content"})

        output_nodes = self.splitter([node0, node1])

        self.assertEqual(len(output_nodes), 2)
        self.assertEqual(output_nodes[0].value_dict["text"], ["# Header ## Content"])
        self.assertEqual(output_nodes[1].value_dict["text"], ["# Header", "## Content"])

    def test_header_splitter(self):
        markdown_str0 = "# Header\n## Content"
        markdown_str1 = "# Header\n## Content\n# Header 2 ## Content 2"

        result0 = self.splitter.header_splitter(markdown_str0)
        result1 = self.splitter.header_splitter(markdown_str1)

        self.assertEqual(result0, ["# Header", "## Content"])
        self.assertEqual(result1, ["# Header", "## Content", "# Header 2 ## Content 2"])

    def test_header_splitter_with_custom_headers(self):
        markdown_str = " <h1># Header</h1> \n Content"
        headers_to_split_on_list0 = []
        headers_to_split_on_list1 = [("\n", "h2")]
        headers_to_split_on_list2 = [("<h1>", "h1")]

        result = self.splitter.header_splitter(markdown_str, None)
        result0 = self.splitter.header_splitter(markdown_str, headers_to_split_on_list0)
        result1 = self.splitter.header_splitter(markdown_str, headers_to_split_on_list1)
        result2 = self.splitter.header_splitter(markdown_str, headers_to_split_on_list2)

        self.assertEqual(result, ['<h1># Header</h1>\nContent'])  
        self.assertEqual(result0, [])  
        self.assertEqual(result1, [])  
        # self.assertEqual(result2, ['<h1># Header</h1>\nContent'])  

    def test_header_splitter_with_no_headers(self):
        markdown_str = "\nContent with no headers"

        result = self.splitter.header_splitter(markdown_str)

        self.assertEqual(result, ['Content with no headers'])  # No split should occur
