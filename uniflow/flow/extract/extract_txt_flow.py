"""Extract txt flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.txt_op import ExtractTxtOp
from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER
from uniflow.op.extract.split.splitter_factory import SplitterOpsFactory


class ExtractTxtFlow(Flow):
    """Extract txt Flow Class."""

    TAG = EXTRACT

    def __init__(self, splitter: str = PARAGRAPH_SPLITTER) -> None:
        """Extract txt Flow Constructor."""
        super().__init__()
        self._extract_txt_op = ExtractTxtOp(name="extract_txt_op")
        self._split_op = SplitterOpsFactory.get(splitter)

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract txt Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_txt_op(nodes)
        nodes = self._split_op(nodes)
        return nodes
