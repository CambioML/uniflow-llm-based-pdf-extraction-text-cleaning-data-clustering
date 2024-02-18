"""Expand operation module."""

from typing import Any, Callable, Mapping, Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class ExpandOp(Op):
    """Expand Operation."""

    def __init__(
        self, name: str, fn: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ) -> None:
        """Initializes expand operation.

        Args:
            name (str): Name of the expand operation.
            fn (callable): Function to expand.
        """
        super().__init__(name)
        self._fn = fn

    def __call__(self, node: Node) -> Sequence[Node]:
        """Calls expand operation.

        Args:
            node (Node): Input node.

        Returns:
            Sequence[Node]: Output nodes.
        """
        output_nodes = []
        value_dicts = self._fn(node.value_dict)
        for value_dict in value_dicts:
            output_nodes.append(
                Node(name=self.unique_name(), value_dict=value_dict, prev_nodes=[node])
            )
        return output_nodes
