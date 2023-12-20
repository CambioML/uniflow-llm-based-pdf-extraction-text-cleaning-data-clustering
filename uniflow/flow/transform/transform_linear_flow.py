"""Flow class."""
from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node.node import Node
from uniflow.op.basic.linear_op import LinearOp
from uniflow.op.prompt_schema import GuidedPrompt


class TransformLinearFlow(Flow):
    """Linear flow class."""

    TAG = TRANSFORM

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:  # pylint: disable=useless-parent-delegation
        """Initialize LinearFlow class."""
        self._linear_op = LinearOp(name="linear_op")
        super().__init__()

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run LinearFlow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._linear_op(nodes)
