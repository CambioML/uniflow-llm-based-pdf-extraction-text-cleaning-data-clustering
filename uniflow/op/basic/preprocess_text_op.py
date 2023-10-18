"""Preprocess operation for text content input."""
from typing import Any, Mapping
import re
from uniflow.op.basic.linear_op import LinearOp

class PreprocessTextOp(LinearOp):
    """Dataset customized clenaup and split into list of paragraphs"""

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform value dict.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        context = value_dict["context"][:]
        # Lowercase
        context = context.lower()
        # Remove leading/trailing whitespace
        context = context.strip()
        # Remove extra space and tabs while MAINTAINING NEW LINE CHARACTERS
        context = re.sub(r'[ \t]+', ' ', context)
        # Remove HTML tags/markups:
        context = re.compile('<.*?>').sub('', context)
        # split it into paragraphs where there are 2+ consecutive newline characters
        paragraphs = re.split(r'\n{2,}', context)

        return {"paragraphs": paragraphs}
