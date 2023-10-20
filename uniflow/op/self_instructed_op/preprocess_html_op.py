"""Preprocess HTML operation."""
from typing import Any, Mapping
import re
from uniflow.op.basic.linear_op import LinearOp
from cleantext import clean
import uniflow.flow.constants as constants
from langchain.document_loaders import UnstructuredHTMLLoader

class PreprocessHTMLOp(LinearOp):
    """Preprocess HTML operation."""

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform value dict.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """

        html_in = value_dict[constants.HTML_KEY][:]
        loader = UnstructuredHTMLLoader("./"+html_in)
        data = loader.load()
        pages = loader.load_and_split()
        print("html_in: ", html_in)
        print("Preprocess HTML Complete!", pages)

        return {constants.PAGES_KEY: pages}
