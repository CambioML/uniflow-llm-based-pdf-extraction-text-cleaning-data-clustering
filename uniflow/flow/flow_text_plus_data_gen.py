""" Text Plus Data Gen Flow class."""
import logging
from typing import Sequence, Mapping, Any
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.flow.flow_data_gen_text import DataGenTextFlow
from uniflow.flow.flow_data_gen import DataGenFlow

from uniflow.flow.constants import OUTPUT_FILE, INPUT_FILE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextPlusDataGenFlow(Flow):
    """Data generation (from text) plus additional data generation flow class."""

    def __init__(self):
        """Initialize Text Plus Data Gen Flow class."""
        super().__init__()
        logger.info("Initializing TextPlusDataGenFlow...")
        self._data_gen_text_flow = DataGenTextFlow()
        self._data_gen_flow = DataGenFlow()
        logger.info("TextPlusDataGenFlow initialization complete!")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Text Plus Data Gen Flow.

        Args:
            nodes: Sequence of nodes to run.

        Returns:
            Sequence of nodes.
        """
        # Run DataTextGen flow
        logger.info("Starting DataGenTextFlow...")
        data_gen_text_out_nodes = self._data_gen_text_flow.run(nodes)
        logger.info("DataGenTextFlow complete!")

        # Run DataGenFlow
        logger.info("Starting DataGenFlow...")
        for node in data_gen_text_out_nodes:
            node.value_dict[INPUT_FILE] = node.value_dict.pop(OUTPUT_FILE)
        data_gen_out_nodes = self._data_gen_flow.run(data_gen_text_out_nodes)
        logger.info("DataGenFlow complete!")
        return data_gen_out_nodes
