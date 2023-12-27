"""Extract txt flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.md_op import ExtractMarkdownOp, ProcessMarkdownOp


class ExtractMarkdownFlow(Flow):
    """Extract markdown Flow Class."""

    TAG = EXTRACT

    def __init__(self) -> None:
        """Extract markdown Flow Constructor."""
        super().__init__()
        self._extract_md_op = ExtractMarkdownOp(name="extract_md_op")
        self._process_md_op = ProcessMarkdownOp(name="process_md_op")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract markdown Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_md_op(nodes)
        nodes = self._process_md_op(nodes)
        return nodes
