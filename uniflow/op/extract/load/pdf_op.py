"""Model Op Module."""
import copy
import re
from typing import Sequence

from uniflow.node.node import Node
from uniflow.op.model.model_op import LLMDataProcessor
from uniflow.op.op import Op


class ProcessPDFOp(Op):
    """Process PDF Op Class."""

    def __init__(self, name: str, model: LLMDataProcessor) -> None:
        """Process PDF Op Constructor.

        Args:
            name (str): Name of the op.
            model (Model): Model to run.
        """
        super().__init__(name)
        self._model = model

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
            value_dict = self._model.run(value_dict)
            text = value_dict["response"][0]
            text = re.split(r"\s*\n\s*", text.strip())
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes
