import unittest
from uniflow.flow.flow import Flow


class TestFlow(unittest.TestCase):
    def test_flow(self):
        # Test __call__ method
        flow = Flow()
        input_dict = {"a": 1, "b": 2}
        output_dict = flow(input_dict)
        self.assertEqual(output_dict, {"output": [{"a": 1, "b": 2}]})

        # Test root property
        self.assertEqual(flow.root.name, "root")
        self.assertEqual(flow.root.value_dict, input_dict)

        # Test _enter method
        nodes = flow._enter(input_dict)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "root")
        self.assertEqual(nodes[0].value_dict, input_dict)

        # Test _run method
        nodes = flow._run(nodes)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "root")
        self.assertEqual(nodes[0].value_dict, input_dict)

        # Test _exit method
        output_dict = flow._exit(nodes)
        self.assertEqual(output_dict, {"output": [{"a": 1, "b": 2}]})
