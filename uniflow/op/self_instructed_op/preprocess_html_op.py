"""Preprocess HTML operation."""
from typing import Any, Mapping
import re
import logging
from uniflow.op.basic.linear_op import LinearOp
from cleantext import clean
import uniflow.flow.constants as constants
from langchain.document_loaders import UnstructuredHTMLLoader


class PreprocessHTMLOp(LinearOp):
    """Preprocess HTML operation."""

    def __init__(self, name: str):
        """Initialize PreprocessHTMLOp class."""
        super().__init__(name)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform value dict.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        self._logger.info("Starting Preprocess HTML...")
        html_in = value_dict[constants.HTML_KEY][:]
        loader = UnstructuredHTMLLoader("./" + html_in)
        data = loader.load()
        pages = loader.load_and_split()
        self._logger.debug("html_in: %s", html_in)
        self._logger.info("Preprocess HTML Complete!")
        self._logger.debug("Output %s", pages)

        return {constants.PAGES_KEY: pages}
