from typing import Sequence, Callable, Tuple, Any, Mapping
import itertools
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

    def _default_split_func(
        self, value_dict: Mapping[str, Any]
    ) -> Tuple[Mapping, Mapping]:
        """Splits the value_dict into two halves based on indices."""
        n = len(value_dict)
        half = n // 2
        half_1 = dict(itertools.islice(value_dict.items(), 0, half))
        half_2 = dict(itertools.islice(value_dict.items(), half, None))
        return (half_1, half_2)

    def __call__(self, root: Node) -> Sequence[Node]:
        """Expands the root node into two nodes.

        Args:
            root (Node): Input node.

        Returns:
            Sequence[Node]: 2 Output nodes, each is a half of the Input node"""
        value_dict_list = root.value_dict
        expand_1_dict_list, expand_2_dict_list = [], []

        for value_dict in value_dict_list:
            expand_1_dict, expand_2_dict = self.split_func(value_dict)
            expand_1_dict_list.append(expand_1_dict)
            expand_2_dict_list.append(expand_2_dict)

        expand_1 = Node(
            name=self.unique_name(), value_dict=expand_1_dict_list, prev_nodes=[root]
        )
        expand_2 = Node(
            name=self.unique_name(), value_dict=expand_2_dict_list, prev_nodes=[root]
        )

        return expand_1, expand_2
