"""Flow class."""
import copy
import uniflow.flow.constants as constants

from typing import Any, Mapping, Sequence
from uniflow.node.node import Node


class Flow:
    """Flow class."""

    def __init__(self):
        """Initialize Flow class."""
        self._root = None

    def __call__(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run flow.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict."""
        nodes = self._enter(value_dict)
        nodes = self._run(nodes)
        output_dict = self._exit(nodes)
        return output_dict

    @property
    def root(self) -> Node:
        """Return root node.

        Returns:
            Node: Root node.
        """
        return self._root

    def _enter(self, value_dict: Mapping[str, Any]) -> Sequence[Node]:
        """Enter flow.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Sequence[Node]: Nodes.
        """
        value_dict = copy.deepcopy(value_dict)
        self._root = Node(name=constants.ROOT_NAME, value_dict=value_dict)
        return [self._root]

    def _run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Sequence[Node]: Nodes.
        """
        return nodes

    def _exit(self, nodes) -> Mapping[str, Any]:
        """Exit flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        return {
            constants.OUTPUT_NAME: [copy.deepcopy(node.value_dict) for node in nodes]
        }
