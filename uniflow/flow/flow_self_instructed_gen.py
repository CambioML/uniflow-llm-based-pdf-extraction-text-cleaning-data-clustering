"""Self Instructed Flow class."""
from typing import Sequence, Mapping, Any
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.op.self_instructed_op.preprocess_html_op import PreprocessHTMLOp
from uniflow.op.self_instructed_op.preprocess_pdf_op import PreprocessPDFOp
from uniflow.op.self_instructed_op.si_model_inf_op import SIModelInfOp
from uniflow.op.self_instructed_op.data_output_si_op import DataOutSIOp
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SelfInstructedGenFlow(Flow):
    """Data generation from HTML via Self Instructed Fine tuning."""

    PreprocessOp = PreprocessHTMLOp

    def __init__(self):
        """Initialize Self Instructed Gen Flow class."""
        logger.info("Initializing SelfInstructGenFlow...")
        super().__init__()
        self._preprocess_op = self.PreprocessOp("preprocess_op")
        self._data_output_si_op = DataOutSIOp("data_output_si_op")
        logger.info("SelfInstructGenFlow initialization complete!")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Sequence[Node]: Nodes.
        """
        # Preprocessing for a list of datasets?
        preprocess_nodes = self._preprocess_op(nodes)
        si_model_inf_op = SIModelInfOp("si_model_inf_op")
        si_model_nodes = si_model_inf_op(preprocess_nodes)
        node_data_out = self._data_output_si_op(si_model_nodes)
        return node_data_out


class PDFSelfInstructedGenFlow(SelfInstructedGenFlow):
    """Data generation from HTML via Self Instructed Fine tuning."""

    PreprocessOp = PreprocessPDFOp
