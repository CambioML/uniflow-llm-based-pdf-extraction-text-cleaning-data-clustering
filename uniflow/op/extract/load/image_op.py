"""Model Op Module."""
import copy
import re
from typing import Sequence

from uniflow.node import Node
from uniflow.op.model.abs_llm_processor import AbsLLMProcessor
from uniflow.op.op import Op


class ExtractImageOp(Op):
    """Process Image Op Class."""

    def __init__(self, name: str, model: AbsLLMProcessor) -> None:
        """Process PDF Op Constructor.

        Args:
            name (str): Name of the op.
            model (AbsLLMProcessor): Model to run.
        """
        super().__init__(name)
        self._model = model

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Op.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            value_dict = self._model.run(value_dict)
            text = value_dict["response"][0]
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes


class ProcessImageOp(Op):
    """Process Image Op Class."""

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run markdown Op .

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            text = value_dict["text"]
            text = re.sub("\n{3,}", "\n\n", text.strip())
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes
