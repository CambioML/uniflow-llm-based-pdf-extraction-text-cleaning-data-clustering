"""Model Flow Module."""
from typing import Any, Dict, Sequence

from uniflow.constants import EXTRACT
from uniflow.flow import Flow
from uniflow.model.model import PreprocessModel
from uniflow.node.node import Node
from uniflow.op.extract.pdf_op import ProcessPDFOp
from uniflow.schema import GuidedPrompt


class ExtractPDFFlow(Flow):
    """Extract PDF Flow Class."""

    TAG = EXTRACT

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._process_pdf_op = ProcessPDFOp(
            name="process_pdf_op",
            model=PreprocessModel(
                guided_prompt_template=guided_prompt_template,
                model_config=model_config,
            ),
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._process_pdf_op(nodes)
