""" Text Plus Data Gen Flow class."""
from typing import Sequence, Mapping, Any
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.flow.flow_data_gen_text import DataGenTextFlow
from uniflow.flow.flow_data_gen import DataGenFlow
import uniflow.flow.constants as constants


class TextPlusDataGenFlow(Flow):
    """Data generation (from text) plus additional data generation flow class."""

    def _run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        #Run data text gen flow
        CONTEXT_MAX_PRINT_LEN = 300
        print("Starting DataGenTextFlow!")
        print("Context:\n{0}{1}".format(nodes[0].value_dict[constants.CONTEXT_KEY][:CONTEXT_MAX_PRINT_LEN], "...[truncated]" if len(nodes[0].value_dict['context']) > CONTEXT_MAX_PRINT_LEN else ""))
        data_gen_text_flow = DataGenTextFlow()
        text_output_dict = data_gen_text_flow(nodes[0].value_dict)
        print('DataGenTextFlow complete!')

        #Run data gen flow
        MAX_QA_PAIRS = 10
        qaa_input = text_output_dict['output'][0]['QApair_df'][:MAX_QA_PAIRS]
        print(f"Starting DataGenFlow!\nQApair_df:\n{qaa_input}")
        data_gen_flow = DataGenFlow()
        data_gen_input_dict = {constants.QAA_KEY: qaa_input}
        self._node_data_out = data_gen_flow(data_gen_input_dict)

    def _exit(self, nodes) -> Mapping[str, Any]:
        """Exit flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        return self._node_data_out
