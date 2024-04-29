"""Splitter Factory Module."""

from typing import Dict, List

from uniflow.op.extract.split.constants import (
    MARKDOWN_HEADER_SPLITTER,
    PARAGRAPH_SPLITTER,
    RECURSIVE_CHARACTER_SPLITTER,
)
from uniflow.op.extract.split.markdown_header_splitter import MarkdownHeaderSplitter
from uniflow.op.extract.split.pattern_splitter_op import PatternSplitter
from uniflow.op.extract.split.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)


class SplitterOpsFactory:
    """Splitter Ops Factory Class."""

    SPLITTERS = {
        PARAGRAPH_SPLITTER: PatternSplitter,
        MARKDOWN_HEADER_SPLITTER: MarkdownHeaderSplitter,
        RECURSIVE_CHARACTER_SPLITTER: RecursiveCharacterSplitter,
    }

    @classmethod
    def get(cls, config: dict) -> "Op":  # noqa: F821
        """Get splitter.

        Args:
            config (dict): Splitter Config to use.

        Returns:
            Op: Splitter Class.

        Raises:
            ValueError: If no splitter registered under the name.
        """
        splitter_cls = cls.SPLITTERS.get(config["splitter_func"])
        if not splitter_cls:
            raise ValueError(
                f"No splitter registered under '{config['splitter_func']}'"
            )
        return splitter_cls(config)

    @classmethod
    def list(cls) -> Dict[str, List[str]]:
        """List all registered splitters.

        Returns:
            List[str]: List of registered splitters names.
        """
        return list(SplitterOpsFactory.SPLITTERS.keys())
