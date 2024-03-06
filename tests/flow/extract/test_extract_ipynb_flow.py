import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.extract.extract_ipynb_flow import ExtractIpynbFlow
from uniflow.node import Node


class TestExtractIpynbFlow(unittest.TestCase):
    @patch("uniflow.flow.extract.extract_ipynb_flow.ExtractIpynbOp")
    @patch("uniflow.flow.extract.extract_ipynb_flow.ProcessIpynbOp")
    def setUp(self, mock_process_ipynb_op, mock_extract_ipynb_op):
        self.mock_process_ipynb_op = mock_process_ipynb_op
        self.mock_extract_ipynb_op = mock_extract_ipynb_op
        self.extract_ipynb_flow = ExtractIpynbFlow()

    def test_init(self):
        self.mock_extract_ipynb_op.assert_called_once_with(name="extract_ipynb_op")
        self.mock_process_ipynb_op.assert_called_once_with(name="process_ipynb_op")

    def test_run(self):
        # arrange
        nodes = [
            Node(name="node1", value_dict={"filename": "filepath"}),
            Node(name="node2", value_dict={"filename": "filepath"}),
        ]

        self.mock_process_ipynb_op.return_value.return_value = MagicMock()
        self.mock_extract_ipynb_op.return_value.return_value = MagicMock()
        result = self.extract_ipynb_flow.run(nodes)

        self.mock_extract_ipynb_op.return_value.assert_called_once_with(nodes)
        self.mock_process_ipynb_op.return_value.assert_called_once_with(
            self.mock_extract_ipynb_op.return_value.return_value
        )
        self.assertEqual(result, self.mock_process_ipynb_op.return_value.return_value)


if __name__ == "__main__":
    unittest.main()
