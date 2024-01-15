from typing import Sequence
from random import random

from uniflow.node import Node
from uniflow.op.op import Op

class ExpandOp(Op):
    """Expand operation class."""

    _func = None

    def __int__(
        self,
        name: str,
        func = None,
    ) -> None:
        """Expand Op Constructor.

        Args:
            name (str): Name of the op.
            func (Optional): Function to split root.
        """
        super().__init__(self, name=name)
        self._func = func

    def __call__(self, root: Node) -> Sequence[Node]:
        """Call expand operation.

        Args:
            root (Node): Input node.

        Returns:
            Sequence[Node]: Output nodes.
        """

        node_1_value = []
        node_2_value = []
        root_value = root.value_dict
        if len(root_value) > 0:
            if self._func is not None:
                node_1_value, node_2_value = self._func(root)
            else:
                split_idx = len(root_value) // 2
                node_1_value = dict(list(root_value.items())[:split_idx])
                node_2_value = dict(list(root_value.items())[split_idx:])

        node_1 = Node(name="Expand_1", value_dict=node_1_value, prev_nodes=[root])
        node_2 = Node(name="Expand_2", value_dict=node_2_value, prev_nodes=[root])

        return [node_1, node_2]
