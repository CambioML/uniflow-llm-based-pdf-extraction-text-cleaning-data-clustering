"""Pattern splitter op."""

import copy
import re
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class PatternSplitter(Op):
    """Pattern Splitter Op Class"""

    default_separators = "\n\n|\n"
    default_min_chunk_size = 1

    def __init__(self, splitterConfig: dict, name: str = "paragraph_split_op") -> None:
        """Pattern Splitter Op Constructor

        Args:
            splitterConfig (dict): Splitter Config to use.
            name (str): Name of the op.
        """
        super().__init__(name)
        self._splitter_config = splitterConfig
        self._separators = (
            "separators" in splitterConfig and splitterConfig["separators"]
        ) or self.default_separators
        self._min_chunk_size = (
            "min_chunk_size" in splitterConfig and splitterConfig["min_chunk_size"]
        ) or self.default_min_chunk_size

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
            text = re.split(self._separators, text)
            text = [p for p in text if len(p) > self._min_chunk_size]
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )

        return output_nodes
