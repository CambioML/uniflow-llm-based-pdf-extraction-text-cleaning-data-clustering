""" Text Plus Data Gen Flow class."""
from typing import Sequence, Mapping, Any
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.flow.flow_data_gen_text import DataGenTextFlow
from uniflow.flow.flow_data_gen import DataGenFlow
import uniflow.flow.constants as constants


class TextPlusDataGenFlow(Flow):
    """Data generation (from text) plus additional data generation flow class."""

    def __init__(self):
        """Initialize Text Plus Data Gen Flow class."""
        super().__init__()
        self._data_gen_text_flow = DataGenTextFlow()
        self._data_gen_flow = DataGenFlow()

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Text Plus Data Gen Flow.

        Args:
            nodes: Sequence of nodes to run.

        Returns:
            Sequence of nodes.
        """
        # Run DataTextGen flow
        print("Starting DataGenTextFlow...")
        data_gen_text_out_nodes = self._data_gen_text_flow.run(nodes)
        print("DataGenTextFlow complete!")

        # Run DataGenFlow
        print("Starting DataGenFlow...")
        data_gen_out_nodes = self._data_gen_flow.run(data_gen_text_out_nodes)
        print("DataGenFlow complete!")
        return data_gen_out_nodes
