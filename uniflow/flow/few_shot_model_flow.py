"""Few shot model flow."""
from typing import Any, Dict, Sequence

from uniflow.flow.flow import Flow
from uniflow.model.model import FewShotModel
from uniflow.node.node import Node
from uniflow.op.model.model_op import ModelOp


class FewShotModelFlow(Flow):
    """Few Shot Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        few_shot_template: Dict[str, Any],
        model_config: Dict[str, Any],
    ) -> None:
        """Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()

        self._model_op = ModelOp(
            name="few_shot_model_op",
            model=FewShotModel(
                model_server=model_server,
                few_shot_template=few_shot_template,
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
