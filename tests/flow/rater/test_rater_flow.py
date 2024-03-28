import unittest
from unittest.mock import MagicMock, patch
from uniflow.flow.rater.rater_flow import RaterFlow
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate

class TestRaterFlow(unittest.TestCase):
    @patch('uniflow.flow.rater.rater_flow.HuggingfaceJsonFormattedLLMRater')
    @patch('uniflow.flow.rater.rater_flow.OpenAIJsonFormattedLLMRater')
    @patch('uniflow.flow.rater.rater_flow.LmRaterModel')
    @patch('uniflow.flow.rater.rater_flow.ModelOp')
    def setUp(self, mock_model_op, mock_lm_rater_model, mock_openai_rater_model, mock_huggingface_rater_model):
        self.mock_model_op = mock_model_op
        self.mock_lm_rater_model = mock_lm_rater_model
        self.mock_openai_rater_model = mock_openai_rater_model
        self.mock_huggingface_rater_model = mock_huggingface_rater_model
        self.mock_lm_rater_model.return_value = MagicMock()
        self.prompt_template = PromptTemplate(instruction="instruction", few_shot_prompt=[{}])
        self.model_config_openai = {"response_format": {"type": "json_object"}, "model_server": "OpenAI"}
        self.model_config_huggingface = {"response_format": {"type": "json_object"}, "model_server": "HuggingFace"}
        self.model_config_rater = {"response_format": {"type": "other"}, "model_server": "open_ai"}
        self.label2score = {"label1": 1.0, "label2": 2.0}
        self.rater_flow_openai = RaterFlow(self.prompt_template, self.model_config_openai, self.label2score)
        self.rater_flow_huggingface = RaterFlow(self.prompt_template, self.model_config_huggingface, self.label2score)
        self.rater_flow_rater = RaterFlow(self.prompt_template, self.model_config_rater, self.label2score)

    def test_init_json_openAI(self):
        self.mock_model_op.assert_called()
        self.mock_openai_rater_model.assert_called_once_with(prompt_template=self.prompt_template, model_config=self.model_config_openai, label2score=self.label2score)

    def test_init_json_huggingface(self):
        self.mock_model_op.assert_called()
        self.mock_huggingface_rater_model.assert_called_once_with(prompt_template=self.prompt_template, model_config=self.model_config_huggingface, label2score=self.label2score)
    
    def test_init_not_json(self):
        self.mock_model_op.assert_called()
        self.mock_lm_rater_model.assert_called_once_with(prompt_template=self.prompt_template, model_config=self.model_config_rater, label2score=self.label2score)
    
    def test_run_openai(self):
        nodes = [Node(name="node1", value_dict={"a": 1}), Node(name="node2", value_dict={"b": 2})]
        self.rater_flow_openai.run(nodes)
        self.mock_model_op.return_value.assert_called_once_with(nodes)
    
    def test_run_huggingface(self):
        nodes = [Node(name="node1", value_dict={"a": 1}), Node(name="node2", value_dict={"b": 2})]
        self.rater_flow_huggingface.run(nodes)
        self.mock_model_op.return_value.assert_called_once_with(nodes)
    
    def test_run_not_json(self):
        nodes = [Node(name="node1", value_dict={"a": 1}), Node(name="node2", value_dict={"b": 2})]
        self.rater_flow_rater.run(nodes)
        self.mock_model_op.return_value.assert_called_once_with(nodes)

if __name__ == '__main__':
    unittest.main()