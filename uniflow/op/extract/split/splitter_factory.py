"""Splitter Factory Module."""
from typing import Dict, List

from uniflow.op.extract.split.constants import MARKDOWN_HEADER_SPLITTER, PARAGRAPH_SPLITTER
from uniflow.op.extract.split.markdown_header_splitter import MarkdownHeaderSplitter
from uniflow.op.extract.split.pattern_splitter_op import PatternSplitter


class SplitterOpsFactory:
    """Splitter Ops Factory Class."""

    SPLITTERS = {
        PARAGRAPH_SPLITTER: PatternSplitter(name="paragraph_split_op", splitter="\n\n"),
        MARKDOWN_HEADER_SPLITTER: MarkdownHeaderSplitter(name="markdown_header_split_op"),
    }

    @staticmethod
    def get(name: str) -> "Op":  # noqa: F821
        """Get splitter.

        Args:
            name (str): Splitter name.

        Returns:
            Op: Splitter Class.

        Raises:
            ValueError: If no splitter registered under the name.
        """
        splitter_cls = SplitterOpsFactory.SPLITTERS.get(name)
        if not splitter_cls:
            raise ValueError(f"No splitter registered under '{name}'")
        return splitter_cls

    @classmethod
    def list(cls) -> Dict[str, List[str]]:
        """List all registered splitters.

        Returns:
            List[str]: List of registered splitters names.
        """
        return list(SplitterOpsFactory.SPLITTERS.keys())
