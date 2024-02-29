"""Model Flow Module."""

from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.mm.model import MmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import PromptTemplate


class GoogleMultiModalModelFlow(Flow):
    """Google MultiModal Model Flow Class."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Google Model Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="google_mm_model_op",
            model=MmModel(
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


class TransformGoogleMultiModalModelFlow(GoogleMultiModalModelFlow):
    """Transform Google Flow Class."""

    TAG = TRANSFORM
