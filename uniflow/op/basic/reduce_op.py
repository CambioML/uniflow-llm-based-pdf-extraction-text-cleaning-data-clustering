from typing import Sequence, Callable, Mapping, Any, Sequence
from uniflow.node import Node
from uniflow.op.op import Op

class ReduceOp(Op):
    """Op class for reducing two nodes into one."""

    def __init__(
        self,
        name: str,
        merge_func: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] = None,
    ) -> None:
        """Constructor for ReduceOp.

        Args:
            name: Name of the op.
            merge_func: Optional function to configure how to merge the nodes.
                Defaults to simple dictionary merging.
        """
        super().__init__(name)
        self.merge_func = merge_func or self._default_merge_func

    def _default_merge_func(self, value_dict1: Sequence[Mapping[str, Any]], value_dict2: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        """Merges two value_dicts by simply updating one with the other."""
        len1, len2 = len(value_dict1), len(value_dict2)
        merged_dict_list = []

        if len1 != len2:
            raise ValueError(f"Cannot merge value_dict list of different length {len1} {len2}")
        for i in range(len1):
            merged_dict = {}
            key_list1 = list(value_dict1[i].keys())
            key_list2 = list(value_dict2[i].keys())
            if len(key_list1) < len(key_list2):
                key_list1, key_list2 = key_list2, key_list1
            min_length = len(key_list2)
            for j in range(min_length):
                key1 = key_list1[j]
                value1 = value_dict1[i][key1]
                key2 = key_list2[j]
                value2 = value_dict2[i][key2]
                merged_key = f"{key1} {key2}"
                merged_value = f"{value1} {value2}"
                merged_dict[merged_key] = merged_value
            for i in range(min_length, len(key_list1)):
                key1 = key_list1[i]
                value1 = value_dict1[key1]
                merged_dict[key1] = value1
            merged_dict_list.append(merged_dict)
        return merged_dict_list

    def __call__(self, expand_1: Node, expand_2: Node) -> Sequence[Node]:
        """Reduces expand_1 and expand_2 into a single node."""
        value_dict1 = expand_1.value_dict
        value_dict2 = expand_2.value_dict
        merged_dict = self.merge_func(value_dict1, value_dict2)

        reduce_1 = Node(name=self.unique_name(), value_dict=merged_dict, prev_nodes=[expand_1, expand_2])

        return reduce_1
