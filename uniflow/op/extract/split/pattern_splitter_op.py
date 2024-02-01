import copy
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class PatternSplitter(Op):
    """Pattern Splitter Op Class"""

    def __init__(self, name: str, splitter: str = "\n\n") -> None:
        """Pattern Splitter Op Constructor

        Args:
            name (str): Name of the op.
            splitter (str): Splitter to use. Defaults to "\n\n".
        """
        super().__init__(name)
        self._splitter = splitter

    def __call__(
        self,
        nodes: Sequence[Node],
    ) -> Sequence[Node]:
        """Run Pattern Splitter Op

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running the split.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            text = value_dict["text"]
            text = text.split(self._splitter)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )

        return output_nodes
