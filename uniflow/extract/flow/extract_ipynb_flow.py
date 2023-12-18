"""Extract txt flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow import Flow
from uniflow.node.node import Node
from uniflow.op.extract.ipynb_op import ExtractIpynbOp


class ExtractIpynbFlow(Flow):
    """Extract txt Flow Class."""

    TAG = EXTRACT

    def __init__(self) -> None:
        """Extract txt Flow Constructor."""
        super().__init__()
        self._extract_ipynb_op = ExtractIpynbOp(name="extract_ipynb_op")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract txt Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_ipynb_op(nodes)
        return nodes
