"""Extract md flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.txt_op import ExtractTxtOp
from uniflow.op.extract.split.constants import MARKDOWN_HEADER_SPLITTER
from uniflow.op.extract.split.splitter_factory import SplitterOpsFactory


class ExtractMarkdownFlow(Flow):
    """Extract Markdown Flow Class."""

    TAG = EXTRACT

    def __init__(self, splitter: str = MARKDOWN_HEADER_SPLITTER) -> None:
        """Extract md Flow Constructor."""
        super().__init__()
        self._extract_md_op = ExtractTxtOp(name="extract_md_op")
        self._split_md_op = SplitterOpsFactory.get(splitter)

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract md Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_md_op(nodes)
        nodes = self._split_md_op(nodes)
        return nodes
