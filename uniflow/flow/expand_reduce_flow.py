from uniflow.flow.flow import Flow
from uniflow.constants import EXPAND_REDUCE
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.reduce_op import ReduceOp
from uniflow.node import Node
from uniflow.db.store_node import store_node, store_edge


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
        # for root, nodes are more likely already in the db
        store_node(node=node)

        expand_nodes = self._expand_op(node)
        for expand_node in expand_nodes:
            store_node(node=expand_node)
            store_edge(node=node, next=expand_node)

        reduce_node = self._reduce_op(expand_1=expand_nodes[0], expand_2=expand_nodes[1])
        store_node(reduce_node)
        store_edge(node=expand_nodes[0], next=reduce_node)
        store_edge(node=expand_nodes[1], next=reduce_node)

        return reduce_node
