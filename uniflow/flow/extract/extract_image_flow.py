"""Extract Image Flow."""

from typing import Any, Dict, Sequence

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.extract.load.image_op import ExtractImageOp, ProcessImageOp
from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER
from uniflow.op.extract.split.splitter_factory import SplitterOpsFactory
from uniflow.op.model.llm_preprocessor import LLMDataPreprocessor


class ExtractImageFlow(Flow):
    """Extract Image Flow Class."""

    TAG = EXTRACT

    def __init__(
        self,
        model_config: Dict[str, Any],
        splitter: str = PARAGRAPH_SPLITTER,
    ) -> None:
        """Extract Image Flow Constructor.

        Args:
            model_config (Dict[str, Any]): Model config.
            splitter (str): Splitter to use. Defaults to "".
        """
        super().__init__()
        self._extract_image_op = ExtractImageOp(
            name="extract_image_op",
            model=LLMDataPreprocessor(
                model_config=model_config,
            ),
        )
        self._process_image_op = ProcessImageOp(name="process_image_op")
        self._split_op = SplitterOpsFactory.get(splitter)

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        nodes = self._extract_image_op(nodes)
        nodes = self._process_image_op(nodes)
        nodes = self._split_op(nodes)
        return nodes
