"""Flow class."""
import copy
import uniflow.flow.constants as constants
import logging

from typing import Any, Mapping, Sequence
from uniflow.node.node import Node
from uniflow.op.basic.linear_op import LinearOp
import uniflow.op.utils as utils


class Flow:
    """Flow class."""

    def __init__(self):
        """Initialize Flow class."""
        linear_op_name = "linear_op"
        counter = 0
        for item in utils.OPS_NAME:
            if "/linear_op" in item:
                counter += 1
        if counter > 0:
            linear_op_name = f"linear_op_{counter}"
        self._linear_op = LinearOp(linear_op_name)
        self._root = None
        logging.basicConfig(format="%(levelname)s [%(module)s]: %(message)s")

    def __call__(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run flow.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict."""
        nodes = self._enter(value_dict)
        nodes = self.run(nodes)
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

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Sequence[Node]: Nodes.
        """
        return self._linear_op(nodes)

    def _exit(self, nodes) -> Mapping[str, Any]:
        """Exit flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        return {
            constants.OUTPUT_NAME: [copy.deepcopy(node.value_dict) for node in nodes],
            constants.ROOT_NAME: self._root,
        }
