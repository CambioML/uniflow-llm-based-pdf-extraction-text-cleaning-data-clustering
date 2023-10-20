"""
This module contains unit tests for the DataGenFlow class in the
uniflow.flow.flow_data_gen module.

The tests in this module use the unittest framework and the mock
library to test the behavior of the DataGenFlow class.
NOTE: To be compatible with the output csv file path, you need to create the folder [path to the root directory of the project]/data/output.
"""

import unittest
from unittest.mock import patch
import pandas as pd
from uniflow.flow.flow_data_gen_text import DataGenTextFlow
from uniflow.op.basic.lmqg_op import LMQGOp

class TestDataGen(unittest.TestCase):
    """Test data generation flow."""
    # mock the lmqg operation as outputting the same as its input plus 'yes'
    @patch.object(LMQGOp, "_transform", side_effect=lambda x: {"qaa_raw": [[(x['paragraphs'][0], "Yes")]]})
    def test_data_text_gen(self, mock_transform):
        # Load dummy data
        context = "With limited exceptions, all EB-2 and EB-3 green card applications require that the employer obtain a Labor Certification from the U.S. Department of Labor. For petitions requiring this step, the Labor Certification process is often the hardest and most arduous step. Prior to being able to file the Labor Certification application, the employer must obtain a prevailing wage from the Department of Labor and prove that there are no minimally qualified U.S. workers available for the positions through the completion of a competitive recruitment process."        
        # Questions and answers as root node
        input_dict = {"context": context}
        # prepared mocked output of lmgq operation using its expected input
        expected_question = 'with limited exceptions, all eb-2 and eb-3 green card applications require that the employer obtain a labor certification from the u.s. department of labor. for petitions requiring this step, the labor certification process is often the hardest and most arduous step. prior to being able to file the labor certification application, the employer must obtain a prevailing wage from the department of labor and prove that there are no minimally qualified u.s. workers available for the positions through the completion of a competitive recruitment process.'
        expected_anser = 'Yes'

        # create flow
        flow = DataGenTextFlow()
        # run flow
        output_dict = flow(input_dict)

        # Assert that the output_dict contains the correct number of outputs
        self.assertEqual(len(output_dict["output"]), 1)

        expected_output = {
            "QApair_df": pd.DataFrame({
                "Question": [expected_question],
                "Answer": [expected_anser]
            })
        }
        # Assert that the output_dict contains the expected augmented questions/answers
        self.assertEqual((output_dict["output"][0]["QApair_df"] == expected_output["QApair_df"]).all().all(), True)
        # Test root property
        self.assertEqual(flow.root.name, "root")
        # Test root value_dict
        assert (flow.root.value_dict["context"]
                == input_dict["context"])
