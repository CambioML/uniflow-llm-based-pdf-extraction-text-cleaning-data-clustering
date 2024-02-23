"""Transform operation module."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from uniflow.node import Node
from uniflow.op.op import Op


class TransformOp(Op):
    """Transform Operation."""

    def __init__(
        self,
        name: str,
        fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        """Initializes transform operation.

        Args:
            name (str): Name of the transform operation.
            fn (callable): Function to transform.
        """
        super().__init__(name)
        self._fn = fn

    def __call__(self, nodes: Sequence[Tuple[Node, Node]]) -> Sequence[Node]:
        """Calls transform operation.

        Args:
            nodes (Sequence[Node]): Input nodes.

        Returns:
            Sequence[Node]: Output nodes.
        """
        output_nodes = []
        for node in nodes:
            value_dict = self._fn(node.value_dict)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict=value_dict,
                    prev_nodes=[node],
                )
            )
        return output_nodes
