import unittest
from unittest.mock import MagicMock, patch

from uniflow.flow.transform.transform_google_flow import GoogleModelFlow
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestGoogleModelFlow(unittest.TestCase):
    @patch("uniflow.flow.transform.transform_google_flow.LmModel")
    @patch("uniflow.flow.transform.transform_google_flow.ModelOp")
    def setUp(self, mock_model_op, mock_lm_model):
        self.mock_model_op = mock_model_op
        self.mock_lm_model = mock_lm_model
        self.mock_lm_model.return_value = MagicMock()
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )
        self.model_config = {"response_format": {"type": "json_object"}}
        self.google_flow = GoogleModelFlow(self.prompt_template, self.model_config)
        # self.lm_model = LmModel(prompt_template=self.prompt_template, model_config=self.model_config)

    def test_init(self):
        self.mock_model_op.assert_called_once_with(
            name="google_model_op",
            model=self.mock_lm_model.return_value,  # This represents the LmModel instance
        )

    def test_run(self):
        nodes = [
            Node(name="node1", value_dict={"a": 1}),
            Node(name="node2", value_dict={"b": 2}),
        ]
        result = self.google_flow.run(nodes)
        self.mock_model_op.return_value.assert_called_once_with(nodes)
        expected_result = self.mock_model_op.return_value(nodes)
        self.assertEqual(result, expected_result)


# class TestTransformGoogleFlow(TestGoogleModelFlow):
#     def setUp(self):
#         super().setUp()
#         self.transform_google_flow = TransformGoogleFlow(self.prompt_template, self.model_config)

if __name__ == "__main__":
    unittest.main()
