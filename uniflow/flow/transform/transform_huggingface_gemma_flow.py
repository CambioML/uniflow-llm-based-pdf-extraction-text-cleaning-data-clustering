"""Gemma Model Flow Module."""

from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.lm.model import LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import PromptTemplate

class GemmaModelFlow(Flow):
    """Gemma Model Flow Class."""

    def __init__(self, prompt_template: PromptTemplate, model_config: Dict[str, Any]):
        """Gemma Model Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Configuration specific to the Gemma model.
        """
        super().__init__()
        # Use LmModel with potentially updated parameters to fit the Gemma model requirements
        self._model_op = ModelOp(
            name="gemma_model_op",
            # Ensure LmModel is initialized with the necessary Gemma configurations
            model=LmModel(
                prompt_template=prompt_template,
                model_config=model_config,
            ),
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        This method orchestrates the flow's execution, applying the LmModel
        to each node in the sequence.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after being processed by the Gemma model.
        """
        # Pass nodes through the model operation
        return self._model_op(nodes)

class TransformGemmaFlow(GemmaModelFlow):
    """Transform Gemma Flow Class."""

    TAG = TRANSFORM
