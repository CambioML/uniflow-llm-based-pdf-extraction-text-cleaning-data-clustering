"""Classify flow for single point auto-rater."""

from typing import Any, Dict, Sequence

from uniflow.constants import RATER
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.llm_rater import JsonFormattedLLMRater, LLMRater
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt_schema import GuidedPrompt


class RaterFlow(Flow):
    """Rater Classification Flow."""

    TAG = RATER

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
        label2score: Dict[str, float],
    ) -> None:
        """Rater Flow Constructor.

        Args:
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
            label2score (Dict[str, float]): String to score mapping.
        """
        super().__init__()
        if (
            "response_format" in model_config
            and model_config["response_format"]["type"] == "json_object"  # noqa: W503
        ):
            model = JsonFormattedLLMRater(
                guided_prompt_template=guided_prompt_template,
                model_config=model_config,
                label2score=label2score,
            )
        else:
            model = LLMRater(
                guided_prompt_template=guided_prompt_template,
                model_config=model_config,
                label2score=label2score,
            )
        self._model_op = ModelOp(
            name="rater_model_op",
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
