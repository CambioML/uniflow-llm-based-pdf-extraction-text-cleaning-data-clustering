import copy
from typing import Any, Callable, Mapping, Sequence

from uniflow.node import Node
from uniflow.op.op import Op

class ReduceOp(Op):
    """Reduction operation class."""

    def __init__(self, merge_function: Callable):
        """
        Constructor for ReduceOp class.

        Args:
            merge_function (Callable): A function to configure alternative strategies for merging expand_1 and expand_2 into reduce_1.
        """
        super().__init__()
        self.merge_function = merge_function

    # def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    #     """Transform value dict.

    #     Args:
    #         value_dict (Mapping[str, Any]): Input value dict.

    #     Returns:
    #         Mapping[str, Any]: Output value dict.
    #     """
    #     return copy.deepcopy(value_dict)

    def __call__(self, expand_1: Sequence[Node], expand_2: Sequence[Node]) -> Sequence[Node]:
        """Call reduction operation.

        Args:
            Input nodes : expand_1 and expand_2.

        Returns:
            Sequence[Node]: Output nodes (reduce_1).
        """
        min_node_len = min(len(expand_1),len(expand_2))
        reduce_output = []

        for i in range(min_node_len):
            combined_value_dict = self.combine_function(
                expand_1[i].value_dict, expand_2[i].value_dict
            )
            reduce_output.append(
                Node(name = self.unique_name(),
                     value_dict=combined_value_dict,
                     prev_nodes=[expand_1[i],expand_2[i]])
            )
        if len(expand_1) > min_node_len:
            reduce_output.extend(expand_1[min_node_len:])
        elif len(expand_2) > min_node_len:
            reduce_output.extend(expand_2[min_node_len:])



