"""Extract txt op."""

import copy
from typing import Sequence

from uniflow.node import Node
from uniflow.op.extract.load.utils import read_file
from uniflow.op.op import Op


class ExtractTxtOp(Op):
    """Extract txt Op Class."""

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Op.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            text = read_file(value_dict["filename"])
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes
