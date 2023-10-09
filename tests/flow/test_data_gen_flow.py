import os
import pandas as pd
from uniflow.flow.flow_data_gen import DataGenFlow
import unittest

class TestFlow(unittest.TestCase):
    def test_flow(self):
        # Test __call__ method
        flow = DataGenFlow()

        # Load data
        current_directory = os.getcwd()
        qaa = pd.read_csv(f"{current_directory}/tests/data/raw_input/qd_immigration.csv", encoding = "utf8")
        # Questions and answers as root node
        input_dict = {"qaa": qaa[:3]}
        output_dict = flow(input_dict)
        # self.assertEqual(output_dict, {"output": [{"a": 1, "b": 2}]})

        # Test root property
        self.assertEqual(flow.root.name, "root")
        # Test root value_dict
        assert (flow.root.value_dict["qaa"]["Question"]==input_dict["qaa"]["Question"]).all()
        assert (flow.root.value_dict["qaa"]["Answer"]==input_dict["qaa"]["Answer"]).all()
        # Test _enter method
        nodes = flow._enter(input_dict)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "root")
        assert (nodes[0].value_dict["qaa"]["Question"]==input_dict["qaa"]["Question"]).all()
        assert (nodes[0].value_dict["qaa"]["Answer"]==input_dict["qaa"]["Answer"]).all()

        # Test _run method
        nodes = flow._run(nodes)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "data_out_op:1")
        breakpoint()
        self.assertEqual(len(nodes[0].value_dict["QApair_df"]), len(input_dict["qaa"])*3)

        # Test _exit method
        output_dict = flow._exit(nodes)
        breakpoint()
        self.assertEqual((len(output_dict["output"]) + len(output_dict["error_list"])) == len(input_dict["qaa"])*3)