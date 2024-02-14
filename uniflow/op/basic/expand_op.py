import copy
from typing import Any, Mapping, Sequence

from uniflow.node import Node
from uniflow.op.op import Op

class ExpandOp(Op):
    """accept a root node as input and produce two nodes as output"""
    # def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    #     """Transform value dict.

    #     Args:
    #         value_dict (Mapping[str, Any]): Input value dict.

    #     Returns:
    #         Mapping[str, Any]: Output value dict.
    #     """
    #     return copy.deepcopy(value_dict)

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Call linear operation.

        Args:
            nodes (Sequence[Node]): Input nodes.

        Returns:
            Sequence[Node]: Output nodes.
        """
        if len(nodes) == 0:
            raise ValueError("nodes are null.")

        node_len = len(nodes)//2
        expand_1_nodes = []
        expand_2_nodes = []
        index = 0
        for node in nodes:
            if index < node_len:
                value_dict = self._transform(node.value_dict)
                expand_1_nodes.append(
                    Node(name=self.unique_name()),
                    value_dict=node.value_dict, prev_nodes=[node]
                )
            else:
                # value_dict = self._transform(node.value_dict)
                expand_2_nodes.append(
                    Node(name=self.unique_name()),
                    value_dict=value_dict, prev_nodes=[node]
                )
            index = index + 1

        return [expand_1_nodes, expand_2_nodes]