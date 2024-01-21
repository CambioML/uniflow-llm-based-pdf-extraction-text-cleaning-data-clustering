from typing import Sequence, Dict, Any
from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.reduce_op import ReduceOp
from uniflow.op.prompt import PromptTemplate


class ExpandReduceFlow(Flow):
    """Flow that expands a node and then reduces the results."""
    TAG = TRANSFORM

    def __init__(
            self, 
            prompt_template: PromptTemplate,
            model_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        # Instantiate ExpandOp and ReduceOp instances
        self._expand_op = ExpandOp("expand")
        self._reduce_op = ReduceOp("reduce")

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run ExpandReduceFlow
        
        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        # root = self._root  # Access the root node from the Flow class
        output_nodes = []
        for node in nodes:
            expanded_nodes = self._expand_op(node)  # Apply expand_op to the root node
            reduced_node = self._reduce_op(*expanded_nodes)  # Apply reduce_op to the expanded nodes
            output_nodes.append(reduced_node)
        print(output_nodes)
        return output_nodes
