import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.transform.transform_huggingface_flow import (
    HuggingFaceModelFlow,  # Update the import path as necessary
)
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestHuggingFaceModelFlow(unittest.TestCase):
    @patch(
        "uniflow.flow.transform.transform_huggingface_flow.ModelOp"
    )  # Update the import path as necessary
    @patch(
        "uniflow.flow.transform.transform_huggingface_flow.LmModel"
    )  # Update the import path as necessary
    def setUp(self, mock_lm_model, mock_model_op):
        self.mock_model_op = mock_model_op
        self.mock_lm_model = mock_lm_model
        # Mock the return value of LmModel to simulate its behavior without actual instantiation
        self.mock_lm_model.return_value = MagicMock()
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )
        self.model_config = {"response_format": {"type": "json_object"}}
        self.huggingface_flow = HuggingFaceModelFlow(
            self.prompt_template, self.model_config
        )

    def test_init(self):
        # Verify that ModelOp is correctly instantiated with the expected arguments
        self.mock_model_op.assert_called_once_with(
            name="huggingface_model_op", model=self.mock_lm_model.return_value
        )

    def test_run(self):
        # Prepare some test nodes to run through the flow
        nodes = [
            Node(name="node1", value_dict={"a": 1}),
            Node(name="node2", value_dict={"b": 2}),
        ]
        # Assume the ModelOp operation simply returns the nodes it receives
        self.mock_model_op.return_value.return_value = nodes
        # Run the flow with the test nodes
        result = self.huggingface_flow.run(nodes)
        # Verify that the ModelOp mock was called with the test nodes
        self.mock_model_op.return_value.assert_called_once_with(nodes)
        # Verify the result matches the expected outcome
        self.assertEqual(result, nodes)


if __name__ == "__main__":
    unittest.main()
