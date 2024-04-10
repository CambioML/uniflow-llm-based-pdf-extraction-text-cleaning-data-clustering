import unittest

from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER
from uniflow.op.extract.split.pattern_splitter_op import PatternSplitter
from uniflow.op.extract.split.splitter_factory import SplitterOpsFactory


class TestSplitterOpsFactory(unittest.TestCase):
    def test_get_with_valid_config(self):
        config = {"splitter_func": PARAGRAPH_SPLITTER}
        splitter = SplitterOpsFactory.get(config)
        self.assertIsInstance(splitter, PatternSplitter)

    def test_get_with_invalid_config(self):
        config = {"splitter_func": "invalid"}
        with self.assertRaises(ValueError):
            SplitterOpsFactory.get(config)

    def test_get_with_empty_config(self):
        config = {}
        with self.assertRaises(KeyError):
            SplitterOpsFactory.get(config)
