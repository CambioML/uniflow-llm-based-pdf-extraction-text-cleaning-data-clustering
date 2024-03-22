import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.transform.transform_google_multimodal_flow import (
    GoogleMultiModalModelFlow,
)
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestGoogleMultiModalModelFlow(unittest.TestCase):
    @patch("uniflow.flow.transform.transform_google_multimodal_flow.MmModel")
    @patch("uniflow.flow.transform.transform_google_multimodal_flow.ModelOp")
    def setUp(self, mock_model_op, mock_mm_model):
        self.mock_model_op = mock_model_op
        self.mock_mm_model = mock_mm_model
        self.mock_mm_model.return_value = MagicMock()
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )
        self.model_config = {"response_format": {"type": "json_object"}}
        self.google_mm_flow = GoogleMultiModalModelFlow(
            self.prompt_template, self.model_config
        )

    def test_init(self):
        self.mock_model_op.assert_called_once_with(
            name="google_mm_model_op",
            model=self.mock_mm_model.return_value,  # This represents the MmModel instance
        )

    def test_run(self):
        nodes = [
            Node(name="node1", value_dict={"a": 1}),
            Node(name="node2", value_dict={"b": 2}),
        ]
        # Mock the __call__ method of the ModelOp instance to return the nodes directly for simplicity
        self.mock_model_op.return_value.return_value = nodes
        result = self.google_mm_flow.run(nodes)
        self.mock_model_op.return_value.assert_called_once_with(nodes)
        self.assertEqual(result, nodes)


if __name__ == "__main__":
    unittest.main()
