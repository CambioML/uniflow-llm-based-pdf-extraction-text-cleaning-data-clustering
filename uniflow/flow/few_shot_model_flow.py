"""Few shot model flow."""
from typing import Any, Dict, Sequence

from uniflow.flow.model_flow import ModelFlow
from uniflow.model.config import ModelConfig
from uniflow.model.model import FewShotModel
from uniflow.node.node import Node
from uniflow.op.model.model_op import ModelOp

FEW_SHOT_TEMPLATE = {
    "instruction": """Generate one question and its corresponding answer based on the context.
    Following the format of the examples below. The response must include original context, question, and answer.""",
    "examples": [
        {
            "context": """The quick brown fox jumps over the lazy dog.""",
            "question": """What is the color of the fox?""",
            "answer": """brown.""",
        },
        {
            "context": """The quick brown fox jumps over the lazy black dog.""",
            "question": """What is the color of the dog?""",
            "answer": """black.""",
        },
    ],
}


class FewShotModelFlow(ModelFlow):
    """Few Shot Model Flow Class."""

    MODEL_NAME: str = "OpenAIModelServer"
    MODEL_CONFIG: ModelConfig = ModelConfig
    FEW_SHOT_TEMPLATE: Dict[str, Any] = FEW_SHOT_TEMPLATE

    def __init__(self) -> None:
        """Model Flow Constructor."""
        super().__init__()

        self._model_op = ModelOp(
            name="few_shot_model_op",
            model=FewShotModel(
                model_name=self.MODEL_NAME,
                model_config=self.MODEL_CONFIG(),
                few_shot_template=self.FEW_SHOT_TEMPLATE,
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
