import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.extract.extract_html_flow import ExtractHTMLFlow
from uniflow.node import Node
from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER


class TestExtractHTMLFlow(unittest.TestCase):
    @patch("uniflow.flow.extract.extract_html_flow.ExtractHTMLOp")
    @patch("uniflow.flow.extract.extract_html_flow.ProcessHTMLOp")
    @patch("uniflow.flow.extract.extract_html_flow.SplitterOpsFactory")
    def setUp(
        self, mock_splitter_ops_factory, mock_process_html_op, mock_extract_html_op
    ):
        self.mock_splitter_ops_factory = mock_splitter_ops_factory
        self.mock_process_html_op = mock_process_html_op
        self.mock_extract_html_op = mock_extract_html_op
        self.extract_html_flow = ExtractHTMLFlow()

    def test_init(self):
        self.mock_extract_html_op.assert_called_once_with(name="extract_html_op")
        self.mock_process_html_op.assert_called_once_with(name="process_html_op")
        self.mock_splitter_ops_factory.get.assert_called_once_with(PARAGRAPH_SPLITTER)

    def test_run(self):
        # arrange
        nodes = [
            Node(name="node1", value_dict={"filename": "filepath"}),
            Node(name="node2", value_dict={"filename": "filepath"}),
        ]

        self.mock_splitter_ops_factory.get.return_value.return_value = MagicMock()
        self.mock_process_html_op.return_value.return_value = MagicMock()
        self.mock_extract_html_op.return_value.return_value = MagicMock()
        result = self.extract_html_flow.run(nodes)

        self.mock_extract_html_op.return_value.assert_called_once_with(nodes)
        self.mock_process_html_op.return_value.assert_called_once_with(
            self.mock_extract_html_op.return_value.return_value
        )
        self.mock_splitter_ops_factory.get.return_value.assert_called_once_with(
            self.mock_process_html_op.return_value.return_value
        )
        self.assertEqual(
            result, self.mock_splitter_ops_factory.get.return_value.return_value
        )


if __name__ == "__main__":
    unittest.main()
