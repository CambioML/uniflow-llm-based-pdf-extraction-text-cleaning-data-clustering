"""Extract txt flow."""

from typing import Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.aws.s3_op import ExtractS3Op
from uniflow.op.extract.load.txt_op import ExtractTxtOp, ProcessTxtOp


class ExtractTxtFlow(Flow):
    """Extract txt Flow Class."""

    TAG = EXTRACT

    def __init__(self) -> None:
        """Extract txt Flow Constructor."""
        super().__init__()
        self._extract_txt_op = ExtractTxtOp(name="extract_txt_op")
        self._process_txt_op = ProcessTxtOp(name="process_txt_op")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Extract txt Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_txt_op(nodes)
        nodes = self._process_txt_op(nodes)
        return nodes


class ExtractS3TxtFlow(ExtractTxtFlow):
    """Extract S3 txt Flow Class."""

    def __init__(self) -> None:
        """Extract S3 txt Flow Constructor."""
        super().__init__()
        self._extract_txt_op = ExtractS3Op(name="extract_s3_txt_op")
        self._process_txt_op = ProcessTxtOp(name="process_txt_op")
