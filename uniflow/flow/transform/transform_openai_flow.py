"""Model Flow Module."""

from dataclasses import dataclass, field
from typing import Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.config import TransformConfig
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.lm.model import JsonLmModel, LmModel
from uniflow.op.model.model_config import ModelConfig, OpenAIModelConfig
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import Context, PromptTemplate


@dataclass
class TransformOpenAIConfig(TransformConfig):
    """Transform OpenAI Config Class."""

    guided_prompt = PromptTemplate(
        instruction="Generate one question and its corresponding answer based on context. Following the format of the examples below to include the same context, question, and answer in the response.",
        few_shot_prompt=[
            Context(
                context="In 1948, Claude E. Shannon published A Mathematical Theory of\nCommunication (Shannon, 1948) establishing the theory of\ninformation. In his article, Shannon introduced the concept of\ninformation entropy for the first time. We will begin our journey here.",
                question="Who published A Mathematical Theory of Communication in 1948?",
                answer="Claude E. Shannon.",
            )
        ],
    )
    model_config: ModelConfig = OpenAIModelConfig(
        response_format={"type": "json_object"}
    )
    auto_split_long_text: bool = field(default=False)


class OpenAIModelFlow(Flow):
    """OpenAI Model Flow Class."""

    CONFIG = TransformOpenAIConfig(flow_name=classmethod(lambda cls: cls.__name__))

    def __init__(self) -> None:
        """OpenAI Model Flow Constructor."""
        super().__init__()
        print(self.CONFIG.model_config.response_format["type"])
        if self.CONFIG.model_config.response_format["type"] == "json_object":
            model = JsonLmModel(
                prompt_template=self.CONFIG.prompt_template,
                model_config=self.CONFIG.model_config,
            )
        else:
            model = LmModel(
                prompt_template=self.CONFIG.prompt_template,
                model_config=self.CONFIG.model_config,
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
