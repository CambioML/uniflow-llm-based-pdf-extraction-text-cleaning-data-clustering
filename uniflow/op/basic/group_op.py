"""Group operation module."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from uniflow.node import Node
from uniflow.op.op import Op
from uniflow.op.prompt import Context


class GroupOp(Op):
    """Customizable Operation."""

    def __init__(
        self,
        name: str,
        preprocss_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        """Initializes reduce operation.

        Args:
            name (str): Name of the reduce operation.
            fn (callable): Function to reduce.
        """
        super().__init__(name)
        self._fn = fn
        self._preprocess_fn = preprocss_fn

    def __call__(self, nodes_1: Sequence[Node], nodes_2: Sequence[Node]) -> Sequence[Node]:
        """Calls reduce operation.

        Args:
            nodes_1, nodes_2: Input two nodes list.

        Returns:
            Sequence[Node]: Output nodes.
        """

        # examples
        """
        node_a0 = Node("node_a0", ['Introduction'])
        node_a1 = Node("node_a1", ['Introduction'])
        node_a2 = Node("node_a2", ['Abstract'])

        node_b0 = Node("node_b0", ['A paper about life itself'])
        node_b1 = Node("node_b1", ['Life is complicated'])
        node_b2 = Node("node_b2", ['Happy wife, happy life'])

        nodes_1 = [node_a0, node_a1, node_a2]
        nodes_2 = [node_b0, node_b1, node_b2]
        """

        output_nodes = []

        # TODO: handling for missing or uneven sections?
        labels, summaries = zip(*self._preprocess_fn(nodes_1, nodes_2))
        aggregated_summaries = self._fn(labels, summaries)

        # Sorted to make sure 1:1 match for later comparison
        sorted_labels = sorted(aggregated_summaries.keys())
        print("sorted_labels: ", sorted_labels)

        # Get prev nodes
        label_nodes = {label: [] for label in sorted_labels}

        for node in nodes_1:
            label = node.value_dict['response'][0]
            if label in label_nodes:
                label_nodes[label].append(node)

        for label in sorted_labels:
            summary_list = aggregated_summaries[label]
            combined_summary = " ".join(summary_list)
            value_dict = [Context(context=combined_summary)]
            
            prev_nodes = label_nodes[label]

            for node in nodes_2:
                if node.value_dict['response'][0] in summary_list:
                    prev_nodes.append(node)
            
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict=value_dict,
                    prev_nodes=prev_nodes
                )
            )

        return output_nodes