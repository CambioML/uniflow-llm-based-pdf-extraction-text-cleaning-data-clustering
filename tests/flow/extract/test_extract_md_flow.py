import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.extract.extract_md_flow import ExtractMarkdownFlow
from uniflow.node import Node
from uniflow.op.extract.split.constants import MARKDOWN_HEADER_SPLITTER


class TestExtractMarkdownFlow(unittest.TestCase):
    @patch("uniflow.flow.extract.extract_md_flow.ExtractTxtOp")
    @patch("uniflow.flow.extract.extract_md_flow.SplitterOpsFactory")
    def setUp(self, mock_splitter_ops_factory, mock_extract_md_op):
        self.mock_splitter_ops_factory = mock_splitter_ops_factory
        self.mock_extract_md_op = mock_extract_md_op
        self.extract_md_flow = ExtractMarkdownFlow()

    def test_init(self):
        self.mock_extract_md_op.assert_called_once_with(name="extract_md_op")
        self.mock_splitter_ops_factory.get.assert_called_once_with(
            MARKDOWN_HEADER_SPLITTER
        )

    def test_run(self):
        # arrange
        nodes = [
            Node(name="node1", value_dict={"filename": "filepath"}),
            Node(name="node2", value_dict={"filename": "filepath"}),
        ]

        self.mock_splitter_ops_factory.get.return_value.return_value = MagicMock()
        self.mock_extract_md_op.return_value.return_value = MagicMock()
        result = self.extract_md_flow.run(nodes)

        self.mock_extract_md_op.return_value.assert_called_once_with(nodes)
        self.mock_splitter_ops_factory.get.return_value.assert_called_once_with(
            self.mock_extract_md_op.return_value.return_value
        )
        self.assertEqual(
            result, self.mock_splitter_ops_factory.get.return_value.return_value
        )


if __name__ == "__main__":
    unittest.main()
