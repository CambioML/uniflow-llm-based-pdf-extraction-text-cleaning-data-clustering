import unittest

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
from uniflow.op.extract.split.splitter_factory import SplitterOpsFactory


class TestSplitterOpsFactory(unittest.TestCase):
    def setUp(self):
        self.paragraph_splitter = SplitterOpsFactory.get(PARAGRAPH_SPLITTER)
        self.markdown_header_splitter = SplitterOpsFactory.get(MARKDOWN_HEADER_SPLITTER)
        self.recursive_character_splitter = SplitterOpsFactory.get(
            RECURSIVE_CHARACTER_SPLITTER
        )

    def test_get(self):
        self.assertTrue(isinstance(self.paragraph_splitter, PatternSplitter))
        self.assertTrue(
            isinstance(self.markdown_header_splitter, MarkdownHeaderSplitter)
        )
        self.assertTrue(
            isinstance(self.recursive_character_splitter, RecursiveCharacterSplitter)
        )

    def test_get_with_invalid_name(self):
        with self.assertRaises(ValueError):
            SplitterOpsFactory.get("")

    def test_list(self):
        excepted_splitters = [
            PARAGRAPH_SPLITTER,
            MARKDOWN_HEADER_SPLITTER,
            RECURSIVE_CHARACTER_SPLITTER,
        ]

        self.assertEqual(SplitterOpsFactory.list(), excepted_splitters)
