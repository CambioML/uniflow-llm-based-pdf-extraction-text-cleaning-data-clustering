"""Gemma Model Flow Module."""

from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.lm.model_server import GemmaModelServer
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import PromptTemplate

class HuggingFaceGemmaModelFlow(Flow):
    """Gemma Model Flow Class."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Gemma Model Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Gemma model configuration.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="gemma_model_op",
            model=GemmaModelServer(
                prompt_template=prompt_template,
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
        return self._model_op(nodes)

class TransformHuggingFaceGemmaFlow(HuggingFaceGemmaModelFlow):
    """Transform Gemma Flow Class."""

    TAG = TRANSFORM
