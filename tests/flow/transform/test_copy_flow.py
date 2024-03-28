import unittest
from unittest.mock import patch

from uniflow.flow.transform.transform_copy_flow import TransformCopyFlow
from uniflow.node import Node
from uniflow.op.prompt import PromptTemplate


class TestTransformCopyFlow(unittest.TestCase):
    @patch("uniflow.flow.transform.transform_copy_flow.CopyOp")
    def setUp(self, mock_copy_op):
        self.mock_copy_op = mock_copy_op
        self.prompt_template = PromptTemplate(
            instruction="instruction", few_shot_prompt=[{}]
        )
        self.model_config = {"response_format": {"type": "json_object"}}
        self.copy_flow = TransformCopyFlow(self.prompt_template, self.model_config)

    def test_init(self):
        self.mock_copy_op.assert_called_once_with(name="copy_op")

    def test_run(self):
        node1 = Node(name="node1", value_dict={"a": 1})
        result = self.copy_flow.run(node1)
        self.mock_copy_op.return_value.assert_called_once_with(node1)
        expected_result = self.mock_copy_op.return_value(node1)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
