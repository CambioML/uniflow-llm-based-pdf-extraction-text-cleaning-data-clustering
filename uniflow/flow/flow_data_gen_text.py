"""Flow class."""
from typing import Sequence
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.op.pdf_qa_gen.preprocess_file_op import PreprocessFileOp
from uniflow.op.pdf_qa_gen.preprocess_text_op import PreprocessTextOp
from uniflow.op.pdf_qa_gen.lmqg_op import LMQGOp
from uniflow.op.pdf_qa_gen.data_output_text_op import DataOutTextOp


class DataGenTextFlow(Flow):
    """Data generation (from text) flow class."""

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Sequence[Node]: Nodes.
        """
        # Preprocessing for a list of datasets?
        preprocess_file_op = PreprocessFileOp("preprocess_file_op")
        node_preproc_file = preprocess_file_op(nodes)
        preprocess_text_op = PreprocessTextOp("preprocess_text_op")
        node_preproc = preprocess_text_op(node_preproc_file)
        lmqg_op = LMQGOp("lmqg_op")
        node_lmqg = lmqg_op(node_preproc)
        data_out_op = DataOutTextOp("data_text_out_op")
        node_data_out = data_out_op(node_lmqg)
        return node_data_out
