"""Linear operation."""
import copy

from typing import Any, Mapping, Sequence
from uniflow.op.op import Op
from uniflow.node.node import Node


class LinearOp(Op):
    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        return copy.deepcopy(value_dict)

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Call linear operation.

        Args:
            nodes (Sequence[Node]): Input nodes.

        Returns:
            Sequence[Node]: Output nodes.
        """
        output_nodes = []
        for node in nodes:
            value_dict = self._transform(node.value_dict)
            output_nodes.append(
                Node(name=self.unique_name(), value_dict=value_dict, prev_nodes=[node])
            )
        return output_nodes
