"""Reduce operation module."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from uniflow.node import Node
from uniflow.op.op import Op


class ReduceOp(Op):
    """Reduce Operation."""

    def __init__(
        self,
        name: str,
        fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        """Initializes reduce operation.

        Args:
            name (str): Name of the reduce operation.
            fn (callable): Function to reduce.
        """
        super().__init__(name)
        self._fn = fn

    def __call__(self, nodes: Sequence[Tuple[Node, Node]]) -> Sequence[Node]:
        """Calls reduce operation.

        Args:
            nodes (Sequence[Tuple[Node, Node]]): Input nodes tuple.

        Returns:
            Sequence[Node]: Output nodes.
        """
        output_nodes = []
        for node1, node2 in nodes:
            value_dict = self._fn(node1.value_dict, node2.value_dict)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict=value_dict,
                    prev_nodes=[node1, node2],
                )
            )
        return output_nodes
