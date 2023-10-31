"""Process file to text operation"""
from typing import Any, Mapping
import re
import logging
from uniflow.op.basic.linear_op import LinearOp
from cleantext import clean
from uniflow.flow.constants import INPUT_FILE, CONTEXT_KEY
from langchain.document_loaders import UnstructuredHTMLLoader
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PreprocessFileOp(LinearOp):
    """Process HTML to text operation."""

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
        context = None

        if input_file is None:
            raise ValueError("input_file is None")
        if input_file.lower().endswith(".html"):
            html_read = open(input_file, "r")
            soup = BeautifulSoup(html_read, "html.parser")
            context = soup.get_text()
        elif input_file.lower().endswith(".txt"):
            with open(input_file, "r") as file:
                context = file.read()
        else:
            raise ValueError("input_file must be either .html or .txt")

        logger.info("Preprocess HTML Complete!")
        logger.debug("Output %s", context)

        return {CONTEXT_KEY: context}
