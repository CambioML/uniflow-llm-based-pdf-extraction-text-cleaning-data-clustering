"""Flow class."""
import copy
import logging
from typing import Any, Mapping, Sequence

from uniflow.flow import constants
from uniflow.flow.flow_factory import FlowFactory
from uniflow.node.node import Node
from uniflow.op.basic.linear_op import LinearOp


class Flow:
    """Flow class."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        FlowFactory.register(name=cls.__name__, flow_cls=cls)

    def __init__(self) -> None:
        """Initialize Flow class."""
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
        return nodes

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


class LinearFlow(Flow):
    """Linear flow class."""

    def __init__(self) -> None:  # pylint: disable=useless-parent-delegation
        """Initialize LinearFlow class."""
        self._linear_op = LinearOp(name="linear_op")
        super().__init__()

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run LinearFlow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._linear_op(nodes)
