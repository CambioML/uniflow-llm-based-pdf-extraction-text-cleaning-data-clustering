"""
This module contains unit tests for the DataGenFlow class in the
uniflow.flow.flow_data_gen module.

The tests in this module use the unittest framework and the mock
library to test the behavior of the DataGenFlow class.
"""
import unittest
from unittest.mock import patch
import pandas as pd
from uniflow.flow.flow_data_gen import DataGenFlow
from uniflow.op.qa_gen.model_inf_op import ModelInfOp
from uniflow.flow.constants import (
    ANSWER_KEY,
    OUTPUT_NAME,
    QAPAIR_DF_KEY,
    QUESTION_KEY,
    ROOT_NAME
    )

class TestDataGen(unittest.TestCase):
    """Test data generation flow."""
    # @patch("uniflow.op.basic.model_inf_op._transform")
    @patch.object(ModelInfOp, "_transform")
    def test_data_gen(self, mock_transform):
        # Load dummy data
        data_in = {
            QUESTION_KEY: ["How are you?", "Coffee or Tea?", "Iced or cold?"],
            ANSWER_KEY: ["I am fine.", "Coffee.", "Iced."]
            }
        qaa = pd.DataFrame(data_in)

        # Questions and answers as root node
        input_dict = {QAPAIR_DF_KEY: qaa}
        # mock the language model inference as outputting 3 augmented questions/answers
        # for each input by appending "1", "2", "3" to the original question/answer.
        mock_q = [[q+"1", q+"2", q+"3"] for q in data_in[QUESTION_KEY]]
        mock_a = [[a+"1", a+"2", a+"3"] for a in data_in[ANSWER_KEY]]
        # prepared mocked output using the same format as the real output
        qaa_augmented_raw = []
        for i in range(len(mock_q)):
            mock_qa_i = [str({"_question": q, "_answer": a})
                         for q, a in zip(mock_q[i], mock_a[i])]
            mock_qa_i = "\n\n######\n".join(mock_qa_i)
            mock_qa_i = "\n\n######\n" + mock_qa_i
            qaa_augmented_raw.append(mock_qa_i)
        # qaa_augmented_raw_df = pd.DataFrame(qaa_augmented_raw)
        dummy_qa = {"qaa_augmented_raw": qaa_augmented_raw}
        mock_transform.return_value = dummy_qa
        # create flow
        flow = DataGenFlow()
        # run flow
        output_dict = flow(input_dict)

        # Assert that the output_dict contains the correct number of outputs
        self.assertEqual(len(output_dict[OUTPUT_KEY]), 1)

        expected_output = {
            QAPAIR_DF_KEY: pd.DataFrame({
                "_question": [q for ql in mock_q for q in ql],
                "_answer": [a for al in mock_a for a in al]
            }),
            "error_list": []
        }
        # Assert that the output_dict contains the expected augmented questions/answers
        self.assertEqual((output_dict[OUTPUT_KEY][0][QAPAIR_DF_KEY]
                          == expected_output[QAPAIR_DF_KEY]).all().all(), True)
        # Assert that the output_dict contains the expected empty error_list
        self.assertEqual(
            output_dict[OUTPUT_KEY][0]['error_list'], expected_output["error_list"])

        # Test root property
        self.assertEqual(flow.root.name, ROOT_NAME)
        # Test root value_dict
        assert (flow.root.value_dict[QAPAIR_DF_KEY][QUESTION_KEY]
                == input_dict[QAPAIR_DF_KEY][QUESTION_KEY]).all()
        assert (flow.root.value_dict[QAPAIR_DF_KEY][ANSWER_KEY]
                == input_dict[QAPAIR_DF_KEY][ANSWER_KEY]).all()
