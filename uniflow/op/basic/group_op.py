"""Group operation module."""

from typing import Any, Callable, Mapping, Optional, Sequence

from uniflow.node import Node
from uniflow.op.op import Op
from uniflow.op.prompt import Context


class GroupOp(Op):
    """Group Operation."""

    def __init__(
        self,
        name: str,
        fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        given_fixed_labels: Optional[list] = None,
    ) -> None:
        """Initializes group operation.

        Args:
            name (str): Name of the group operation.
            fn (callable): Function to group.
            given_fixed_labels (Optional[list]) : A list of fixed, provided labels to help handle exceptions if there are no content for certain labels
        """
        super().__init__(name)
        self._fn = fn
        self._given_fixed_labels = given_fixed_labels if given_fixed_labels else []

    def __call__(
        self, nodes_1: Sequence[Node], nodes_2: Sequence[Node]
    ) -> Sequence[Node]:
        """Calls group operation.
           Then (fn) function would groub by summaries from nodes_2 based on their labels from nodes_1.
           The result would be a list of nodes where each node's dictionary is a sum of summaries of nodes with same label.
           If given_fixed_labels is provided, labels with no summaries will still be included in the result.

        Args:
            nodes_1 (Sequence[Node]), nodes_2 (Sequence[Node]): Input two nodes.

        Returns:
            Sequence[Node]: Output nodes.
        """
        output_nodes = []

        labels, summaries = zip(
            *[
                (
                    node_label.value_dict[0].context[0],
                    node_summary.value_dict[0].context[0],
                )
                for node_label, node_summary in zip(nodes_1, nodes_2)
            ]
        )

        aggregated_summaries = self._fn(labels, summaries)
        sorted_labels = sorted(aggregated_summaries.keys())

        # Exception handling for missing sections (no summaries are given such label)
        if self._given_fixed_labels:
            for label in self._given_fixed_labels:
                if label not in sorted_labels:
                    sorted_labels.append(label)
            sorted_labels.sort()

        label_nodes = {label: [] for label in sorted_labels}

        for node in nodes_1:
            label = node.value_dict[0].context[0]
            if label in label_nodes:
                label_nodes[label].append(node)

        for label in sorted_labels:
            try:
                summary_list = aggregated_summaries[label]
                combined_summary = " ".join(summary_list)
                value_dict = [Context(context=combined_summary)]

                prev_nodes = label_nodes[label]

                for node in nodes_2:
                    if node.value_dict[0].context[0] in summary_list:
                        prev_nodes.append(node)

                output_nodes.append(
                    Node(
                        name=self.unique_name(),
                        value_dict=value_dict,
                        prev_nodes=prev_nodes,
                    )
                )
            except:  # label with empty summaries
                output_nodes.append(
                    Node(
                        name=self.unique_name(),
                        value_dict=[Context(context="")],
                    )
                )

        return output_nodes
