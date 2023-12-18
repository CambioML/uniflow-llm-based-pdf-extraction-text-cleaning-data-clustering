"""Model Flow Module."""
from typing import Any, Dict, Sequence

from uniflow.constants import EXTRACT
from uniflow.flow import Flow
from uniflow.model.model import PreprocessModel
from uniflow.node.node import Node
from uniflow.op.extract.pdf_op import ExtractPDFOp, ProcessPDFOp


class ExtractPDFFlow(Flow):
    """Extract PDF Flow Class."""

    TAG = EXTRACT

    def __init__(
        self,
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._extract_pdf_op = ExtractPDFOp(
            name="extract_pdf_op",
            model=PreprocessModel(
                guided_prompt_template=None,
                model_config=model_config,
            ),
        )
        self._process_pdf_op = ProcessPDFOp(name="process_pdf_op")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_pdf_op(nodes)
        nodes = self._process_pdf_op(nodes)
        return nodes
