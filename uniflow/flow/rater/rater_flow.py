"""Classify flow for single point auto-rater."""

from typing import Any, Dict, Sequence

from uniflow.constants import RATER
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.llm_rater import (
    HuggingfaceJsonFormattedLLMRater,
    LLMRater,
    OpenAIJsonFormattedLLMRater,
)
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import PromptTemplate


class RaterFlow(Flow):
    """Rater Classification Flow."""

    TAG = RATER

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
        label2score: Dict[str, float],
    ) -> None:
        """Rater Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
            label2score (Dict[str, float]): String to score mapping.
        """
        super().__init__()
        if (
            "response_format" in model_config
            and model_config["response_format"]["type"] == "json_object"  # noqa: W503
        ):
            if "openai" in model_config["model_server"].lower():
                model = OpenAIJsonFormattedLLMRater(
                    prompt_template=prompt_template,
                    model_config=model_config,
                    label2score=label2score,
                )
            else:
                # Huggingface json formatted LLM rater
                # will format the response into a json object
                # after the response is returned from the model server.
                model = HuggingfaceJsonFormattedLLMRater(
                    prompt_template=prompt_template,
                    model_config=model_config,
                    label2score=label2score,
                )
        else:
            model = LLMRater(
                prompt_template=prompt_template,
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
