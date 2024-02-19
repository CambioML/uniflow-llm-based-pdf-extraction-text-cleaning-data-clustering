"""Model Flow Module."""

from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.lm.model import JsonLmModel, LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import PromptTemplate


class OpenAIModelFlow(Flow):
    """OpenAI Model Flow Class."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """OpenAI Model Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        if model_config["response_format"]["type"] == "json_object":
            model = JsonLmModel(
                prompt_template=prompt_template,
                model_config=model_config,
            )
        else:
            model = LmModel(
                prompt_template=prompt_template,
                model_config=model_config,
            )
        self._model_op = ModelOp(
            name="openai_model_op",
            model=model,
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._model_op(nodes)


class TransformOpenAIFlow(OpenAIModelFlow):
    """Transform OpenAI Flow Class."""

    TAG = TRANSFORM
