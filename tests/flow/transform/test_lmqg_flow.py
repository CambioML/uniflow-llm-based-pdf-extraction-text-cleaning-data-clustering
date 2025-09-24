import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.transform.transform_lmqg_flow import TransformLMQGFlow
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestTransformLMQGFlow(unittest.TestCase):
    @patch("uniflow.flow.transform.transform_lmqg_flow.ModelOp")
    @patch("uniflow.flow.transform.transform_lmqg_flow.LmModel")
    def setUp(self, mock_lm_model, mock_model_op):
        self.mock_model_op = mock_model_op
        self.mock_lm_model = mock_lm_model
        # Setup the mock for LmModel to simulate its behavior without actual instantiation
        self.mock_lm_model.return_value = MagicMock()
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )
        self.model_config = {"response_format": {"type": "json_object"}}
        self.lmqg_flow = TransformLMQGFlow(self.prompt_template, self.model_config)

    def test_init(self):
        # Verify that ModelOp is instantiated with the correct arguments
        self.mock_model_op.assert_called_once_with(
            name="lmqg_model_op", model=self.mock_lm_model.return_value
        )

    def test_run(self):
        # Prepare some test nodes to process
        nodes = [
            Node(name="node1", value_dict={"a": 1}),
            Node(name="node2", value_dict={"b": 2}),
        ]
        # Assume the ModelOp operation simply returns the nodes it receives for simplicity
        self.mock_model_op.return_value.return_value = nodes
        # Execute the flow with the test nodes
        result = self.lmqg_flow.run(nodes)
        # Verify that the ModelOp mock was invoked with the test nodes
        self.mock_model_op.return_value.assert_called_once_with(nodes)
        # Check that the result matches the expected output
        self.assertEqual(result, nodes)


if __name__ == "__main__":
    unittest.main()
