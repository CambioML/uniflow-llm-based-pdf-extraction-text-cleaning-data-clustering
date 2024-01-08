"""Flow class."""
from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.copy_op import CopyOp
from uniflow.op.prompt import PromptTemplate


class TransformCopyFlow(Flow):
    """Copy flow class.

    This is a demo flow does nothing but copy the input nodes.
    """

    TAG = TRANSFORM

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:  # pylint: disable=useless-parent-delegation
        """Initialize CopyFlow class."""
        self._copy_op = CopyOp(name="copy_op")
        super().__init__()

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run CopyFlow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._copy_op(nodes)
