"""Model Flow Module."""
from typing import Any, Dict, Sequence

from uniflow.constants import MODEL
from uniflow.flow import Flow
from uniflow.model.model import Model
from uniflow.node.node import Node
from uniflow.op.model.model_op import ModelOp
from uniflow.schema import GuidedPrompt


class HuggingFaceModelFlow(Flow):
    """HuggingFace Model Flow Class."""

    TAG = MODEL

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="huggingface_model_op",
            model=Model(
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
        return self._model_op(nodes)
