"""Extract txt op."""

import copy
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class ExtractIpynbOp(Op):
    """Extract ipynb Op Class."""

    def __init__(self, name: str) -> None:
        try:
            import nbformat  # pylint: disable=import-outside-toplevel
            from nbconvert import (  # pylint: disable=import-outside-toplevel
                MarkdownExporter,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install nbformat and nbconvert to load ipynb file. You can use `pip install nbformat nbconvert` to install them."
            ) from exc

        super().__init__(name)
        self._nbformat = nbformat
        self._markdown_exporter = MarkdownExporter

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
            nb = self._nbformat.read(value_dict["filename"], as_version=4)
            md_exporter = self._markdown_exporter()
            (text, _) = md_exporter.from_notebook_node(nb)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes


class ProcessIpynbOp(Op):
    """Process ipynb Op Class."""

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
            text = value_dict["text"]
            text = text.strip()
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes
