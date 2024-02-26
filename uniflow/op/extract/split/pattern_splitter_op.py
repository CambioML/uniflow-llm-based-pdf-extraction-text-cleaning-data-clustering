"""Pattern splitter op."""

import copy
import re
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class PatternSplitter(Op):
    """Pattern Splitter Op Class"""

    def __init__(
        self, splitterConfig: dict[str, any], name: str = "paragraph_split_op"
    ) -> None:
        """Pattern Splitter Op Constructor

        Args:
            splitterConfig (dict): Splitter Config to use.
            name (str): Name of the op.
        """
        super().__init__(name)
        self._splitter_config = splitterConfig

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
            text = re.split(self._splitter_config["separators"], text)
            text = [p for p in text if len(p) > self._splitter_config["min_chunk_size"]]
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )

        return output_nodes
