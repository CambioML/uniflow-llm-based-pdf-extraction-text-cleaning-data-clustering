"""Model Op Module."""
import copy
from typing import Sequence

from uniflow.model.model import Model
from uniflow.node.node import Node
from uniflow.op.op import Op


class ModelOp(Op):
    """Model Op Class."""

    def __init__(self, name: str, model: Model) -> None:
        """Model Op Constructor.

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
            output_nodes.append(
                Node(name=self.unique_name(), value_dict=value_dict, prev_nodes=[node])
            )
        return output_nodes
