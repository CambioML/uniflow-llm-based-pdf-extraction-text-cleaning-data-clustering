"""Extract HTML flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.html_op import ExtractHTMLOp, ProcessHTMLOp


class ExtractHTMLFlow(Flow):
    """Extract HTML Flow Class."""

    TAG = EXTRACT

    def __init__(self) -> None:
        """Extract HTML Flow Constructor."""
        super().__init__()
        self._extract_html_op = ExtractHTMLOp(name="extract_html_op")
        self._process_html_op = ProcessHTMLOp(name="process_html_op")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract HTML Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_html_op(nodes)
        nodes = self._process_html_op(nodes)
        return nodes
