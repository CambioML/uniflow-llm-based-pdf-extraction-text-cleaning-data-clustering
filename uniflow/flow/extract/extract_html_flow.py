"""Extract HTML flow."""

from typing import Callable, Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.transform_op import TransformOp
from uniflow.op.extract.load.html_op import ExtractHTMLOp, ProcessHTMLOp
from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER
from uniflow.op.extract.split.splitter_factory import SplitterOpsFactory


class ExtractHTMLFlow(Flow):
    """Extract HTML Flow Class."""

    TAG = EXTRACT

    def __init__(
        self,
        post_extract_fn: Callable = None,
        splitter: str = PARAGRAPH_SPLITTER,
    ) -> None:
        """Extract HTML Flow Constructor."""
        super().__init__()
        self._extract_html_op = ExtractHTMLOp(name="extract_html_op")
        self._process_html_op = ProcessHTMLOp(name="process_html_op")
        self._split_op = SplitterOpsFactory.get(splitter)
        self._post_extract_op = None
        if post_extract_fn:
            self._post_extract_op = TransformOp(
                name="post_extract_fn", fn=post_extract_fn
            )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract HTML Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_html_op(nodes)
        nodes = self._process_html_op(nodes)
        nodes = self._split_op(nodes)
        if self._post_extract_op:
            nodes = self._post_extract_op(nodes)
        return nodes
