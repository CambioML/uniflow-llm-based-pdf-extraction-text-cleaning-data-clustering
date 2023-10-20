"""Self Instructed Flow class."""
from typing import Sequence, Mapping, Any
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.op.self_instructed_op.preprocess_html_op import PreprocessHTMLOp
from uniflow.op.self_instructed_op.si_model_inf_op import SIModelInfOp
from uniflow.op.self_instructed_op.data_output_si_op import DataOutSIOp


class SelfInstructedGenFlow(Flow):
    """Data generation from HTML via Self Instructed Fine tuning."""

    def _run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Sequence[Node]: Nodes.
        """
        # Preprocessing for a list of datasets?
        preprocess_html_op = PreprocessHTMLOp("preprocess_html_op")
        preprocess_nodes = preprocess_html_op(nodes)
        si_model_inf_op = SIModelInfOp("si_model_inf_op")
        si_model_nodes = si_model_inf_op(preprocess_nodes)
        data_output_si_op = DataOutSIOp("data_output_si_op")
        node_data_out = data_output_si_op(si_model_nodes)
        return node_data_out
