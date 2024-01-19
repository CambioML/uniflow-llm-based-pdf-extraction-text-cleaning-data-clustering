"""Extract ipynb flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.ipynb_op import ExtractIpynbOp, ProcessIpynbOp


class ExtractIpynbFlow(Flow):
    """Extract ipynb Flow Class."""

    TAG = EXTRACT

    def __init__(self) -> None:
        """Extract ipynb Flow Constructor."""
        super().__init__()
        self._extract_ipynb_op = ExtractIpynbOp(name="extract_ipynb_op")
        self._process_ipynb_op = ProcessIpynbOp(name="process_ipynb_op")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract ipynb Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_ipynb_op(nodes)
        nodes = self._process_ipynb_op(nodes)
        return nodes
