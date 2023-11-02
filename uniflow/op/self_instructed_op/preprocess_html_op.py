"""Preprocess HTML operation."""
from typing import Any, Mapping
import re
import logging
from uniflow.op.basic.linear_op import LinearOp
from cleantext import clean
from uniflow.flow.constants import INPUT_FILE, PAGES_KEY
from langchain.document_loaders import UnstructuredHTMLLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PreprocessHTMLOp(LinearOp):
    """Preprocess HTML operation."""

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform value dict.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        logger.info("Starting Preprocess HTML...")
        input_file = value_dict[INPUT_FILE]
        logger.debug("input_file: %s", input_file)
        loader = UnstructuredHTMLLoader(input_file)
        data = loader.load()
        pages = loader.load_and_split()
        logger.debug("Output %s", pages)
        logger.info("Preprocess HTML Complete!")

        return {PAGES_KEY: pages}
