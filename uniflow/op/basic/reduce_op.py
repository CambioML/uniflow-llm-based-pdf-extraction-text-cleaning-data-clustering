from typing import Literal
from random import shuffle

from uniflow.node import Node
from uniflow.op.op import Op

class ReduceOp(Op):
    """Reduce operation class."""

    _func = None

    def __int__(
        self,
        name: str,
        func = None,
    ) -> None:
        """Reduce Op Constructor.

        Args:
            name (str): Name of the op.
            func (Optional): Function to merge nodes.
        """
        super().__init__(self, name=name)
        self._func = func

    def __call__(self, expand_1: Node, expand_2: Node) -> Node:
        """Call reduce operation.

        Args:
            expand_1 (Node): Input node.
            expand_2 (Node): Input node.

        Returns:
            Node: Output node.
        """
        node_value = []
        if self._func is not None:
            node_value = self._func(expand_1.value_dict, expand_2.value_dict)
        else:
            expand_1_value = list(expand_1.value_dict.items())
            expand_2_value = list(expand_2.value_dict.items())

            # length of the two merged lists is the same
            expand_1_value_len = len(expand_1_value)
            expand_2_value_len = len(expand_2_value)
            smaller_size = min(expand_1_value_len, expand_2_value_len)
            for i in range(smaller_size):
                value_1, value_2 = expand_1_value[i], expand_2_value[i]
                key = f"{value_1[0]} {value_2[0]}"
                value = f"{str(value_1[1])} {str(value_2[1])}"
                node_value.append((key, value))

            # append the rest
            if expand_1_value_len > smaller_size:
                node_value.extend(expand_1_value[smaller_size:])
            if expand_2_value_len > smaller_size:
                node_value.extend(expand_2_value[smaller_size:])

        reduce_node = Node(name="Reduce_1", value_dict=dict(node_value), prev_nodes=[expand_1, expand_2])

        return reduce_node
