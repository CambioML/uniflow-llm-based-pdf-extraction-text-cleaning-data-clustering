"""Preprocess PDF operation."""
from typing import Any, Mapping
import re
import logging
from uniflow.op.basic.linear_op import LinearOp
from cleantext import clean
from uniflow.flow.constants import INPUT_FILE, PAGES_KEY
from langchain.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PreprocessPDFOp(LinearOp):
    """Preprocess PDF operation."""

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform value dict.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        logger.info("Starting Preprocess PDF...")
        input_file = value_dict[INPUT_FILE]
        logger.debug("input_file: %s", input_file)
        loader = PyPDFLoader(input_file)
        data = loader.load()
        pages = loader.load_and_split()
        logger.debug("Output %s", pages)
        logger.info("Preprocess PDF Complete!")

        return {PAGES_KEY: pages}
