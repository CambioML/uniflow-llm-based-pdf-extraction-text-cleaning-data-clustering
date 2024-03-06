import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.extract.extract_txt_flow import ExtractTxtFlow
from uniflow.node import Node
from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER


class TestExtractTxtFlow(unittest.TestCase):
    @patch("uniflow.flow.extract.extract_txt_flow.ExtractTxtOp")
    @patch("uniflow.flow.extract.extract_txt_flow.SplitterOpsFactory")
    def setUp(self, mock_splitter_ops_factory, mock_extract_txt_op):
        self.mock_splitter_ops_factory = mock_splitter_ops_factory
        self.mock_extract_txt_op = mock_extract_txt_op
        self.extract_txt_flow = ExtractTxtFlow()

    def test_init(self):
        self.mock_extract_txt_op.assert_called_once_with(name="extract_txt_op")
        self.mock_splitter_ops_factory.get.assert_called_once_with(PARAGRAPH_SPLITTER)

    def test_run(self):
        # arrange
        nodes = [
            Node(name="node1", value_dict={"filename": "filepath"}),
            Node(name="node2", value_dict={"filename": "filepath"}),
        ]

        self.mock_splitter_ops_factory.get.return_value.return_value = MagicMock()
        self.mock_extract_txt_op.return_value.return_value = MagicMock()
        result = self.extract_txt_flow.run(nodes)

        self.mock_extract_txt_op.return_value.assert_called_once_with(nodes)
        self.mock_splitter_ops_factory.get.return_value.assert_called_once_with(
            self.mock_extract_txt_op.return_value.return_value
        )
        self.assertEqual(
            result, self.mock_splitter_ops_factory.get.return_value.return_value
        )


if __name__ == "__main__":
    unittest.main()
