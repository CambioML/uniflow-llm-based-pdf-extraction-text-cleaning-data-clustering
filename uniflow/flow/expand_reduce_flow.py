from uniflow.flow.flow import Flow
from uniflow.constants import EXPAND_REDUCE
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.reduce_op import ReduceOp
from uniflow.node import Node

class ExpandReduceFlow(Flow):
    """ExpandReduceFlow class.
    Expand and then reduce the input node.
    """

    TAG = EXPAND_REDUCE

    def __init__(self) -> None:
        """Initialize ExpandReduceFlow class."""
        super().__init__()
        self._expand_op = ExpandOp(name="expand_op")
        self._reduce_op = ReduceOp(name="reduce_op")

    def run(self, node: Node) -> Node:
        """Run ExpandReduceFlow.

        Args:
            node (Node): Node to run.

        Returns:
            Node: Node after running.
        """
        nodes = self._expand_op(node)
        return self._reduce_op(expand_1=nodes[0], expand_2=nodes[1])
