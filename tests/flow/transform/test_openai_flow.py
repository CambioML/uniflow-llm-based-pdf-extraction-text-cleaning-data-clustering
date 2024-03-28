import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.transform.transform_openai_flow import OpenAIModelFlow
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestOpenAIModelFlow(unittest.TestCase):
    @patch("uniflow.flow.transform.transform_openai_flow.JsonLmModel")
    @patch("uniflow.flow.transform.transform_openai_flow.ModelOp")
    @patch("uniflow.flow.transform.transform_openai_flow.LmModel")
    def setUp(self, mock_lm_model, mock_model_op, mock_json_model):
        self.mock_model_op = mock_model_op
        self.mock_lm_model = mock_lm_model
        self.mock_json_model = mock_json_model
        self.mock_json_model.return_value = MagicMock()
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )
        self.model_config1 = {"response_format": {"type": "json_object"}}
        self.model_config2 = {"response_format": {"type": "other"}}
        self.azure_flow1 = OpenAIModelFlow(self.prompt_template, self.model_config1)
        self.azure_flow2 = OpenAIModelFlow(self.prompt_template, self.model_config2)

    def test_init(self):
        self.mock_json_model.assert_called_once_with(
            prompt_template=self.prompt_template, model_config=self.model_config1
        )
        self.mock_model_op.assert_called()

    def test_not_json_init(self):
        self.mock_lm_model.assert_called_once_with(
            prompt_template=self.prompt_template, model_config=self.model_config2
        )
        self.mock_model_op.assert_called()

    def test_run(self):
        nodes1 = [
            Node(name="node1", value_dict={"a": 1}),
            Node(name="node2", value_dict={"b": 2}),
        ]
        result = self.azure_flow1.run(nodes1)
        self.mock_model_op.return_value.assert_called_once_with(nodes1)
        expected_result = self.mock_model_op.return_value(nodes1)
        self.assertEqual(result, expected_result)
        # self.assertEqual(result, self.mock_model_op.return_value(node1))

    def test_not_json_run(self):
        nodes2 = [
            Node(name="node1", value_dict={"a": 1}),
            Node(name="node2", value_dict={"b": 2}),
        ]
        result = self.azure_flow2.run(nodes2)
        self.mock_model_op.return_value.assert_called_once_with(nodes2)
        self.assertEqual(result, self.mock_model_op.return_value(nodes2))


if __name__ == "__main__":
    unittest.main()
