import unittest
from unittest.mock import patch

from uniflow.flow.transform.transform_azure_openai_flow import AzureOpenAIModelFlow
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestAzureOpenAIModelFlow(unittest.TestCase):
    @patch("uniflow.flow.transform.transform_azure_openai_flow.ModelOp")
    @patch("uniflow.flow.transform.transform_azure_openai_flow.LmModel")
    @patch("uniflow.flow.transform.transform_azure_openai_flow.JsonLmModel")
    def setUp(self, mock_json_model, mock_lm_model, mock_model_op):
        self.mock_json_model = mock_json_model
        self.mock_lm_model = mock_lm_model
        self.mock_model_op = mock_model_op

        self.prompt_template0 = None
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )

        self.model_config0 = None
        self.model_config1 = {"response_format": {"type": "json_object"}}
        self.model_config2 = {"response_format": {"type": "other"}}
        self.azure_flow1 = AzureOpenAIModelFlow(
            self.prompt_template, self.model_config1
        )
        self.azure_flow2 = AzureOpenAIModelFlow(
            self.prompt_template, self.model_config2
        )

    def test_prompt_template_none(self):
        """Test AzureOpenAIModelFlow initialization with prompt_template=None."""
        model_config = self.model_config1  # Mocked model config

        with self.assertRaises(ValueError):
            AzureOpenAIModelFlow(prompt_template=None, model_config=model_config)

    def test_model_config_none(self):
        """Test AzureOpenAIModelFlow initialization with model_config=None."""
        prompt_template = self.prompt_template  # Mocked or real prompt template

        with self.assertRaises(ValueError):
            AzureOpenAIModelFlow(prompt_template=prompt_template, model_config=None)

    def test_init_success(self):
        self.mock_json_model.assert_called_once_with(
            prompt_template=self.prompt_template, model_config=self.model_config1
        )
        self.mock_model_op.assert_called()

    def test_not_json_init(self):
        self.mock_lm_model.assert_called_once_with(
            prompt_template=self.prompt_template, model_config=self.model_config2
        )
        self.mock_model_op.assert_called()

    # def test_call_with_empty_node(self, mock_read_file):
    #         # arrange
    #         nodes = []

    #         # act
    #         output_nodes = self.extract_txt_op(nodes)

    #         # assert
    #         mock_read_file.assert_not_called()
    #         self.assertEqual(len(output_nodes), 0)
    def test_run(self):
        node1 = Node(name="node1", value_dict={"a": 1})
        result = self.azure_flow1.run(node1)
        self.mock_model_op.return_value.assert_called_once_with(node1)
        expected_result = self.mock_model_op.return_value(node1)
        self.assertEqual(result, expected_result)
        # self.assertEqual(result, self.mock_model_op.return_value(node1))

    def test_not_json_run(self):
        node1 = Node(name="node1", value_dict={"a": 1})
        result = self.azure_flow2.run(node1)
        self.mock_model_op.return_value.assert_called_once_with(node1)
        self.assertEqual(result, self.mock_model_op.return_value(node1))


if __name__ == "__main__":
    unittest.main()
