from typing import Sequence, Callable, Tuple, Any, Mapping
from uniflow.node import Node
from uniflow.op.op import Op

class ExpandOp(Op):
    """Op class for expanding a node into two nodes."""

    def __init__(
        self,
        name: str,
        split_func: Callable[[dict], Tuple[dict, dict]] = None,
    ) -> None:
        """Constructor for ExpandOp.

        Args:
            name: Name of the op.
            split_func: Optional function to configure how to split the root node.
                Defaults to splitting based on key-value pair indices.
        """
        super().__init__(name)
        self.split_func = split_func or self._default_split_func

    def _default_split_func(self, value_dict: Mapping[str, Any]) -> Tuple[Mapping, Mapping]:
        """Splits the value_dict into two halves based on indices."""
        n = len(value_dict)
        half = n // 2
        return (dict(value_dict.items()[:half]), dict(value_dict.items()[half:]))

    def __call__(self, root: Node) -> Sequence[Node]:
        """Expands the root node into two nodes."""
        value_dict = self._transform(root.value_dict)
        expand_1_dict, expand_2_dict = self.split_func(value_dict)

        expand_1 = Node(name=self.unique_name(), value_dict=expand_1_dict, prev_nodes=[root])
        expand_2 = Node(name=self.unique_name(), value_dict=expand_2_dict, prev_nodes=[root])

        return expand_1, expand_2
