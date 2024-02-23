"""MultiModal model (MM) model class."""

import copy
from typing import Any, Dict, List

from uniflow.op.model.abs_model import AbsModel
from uniflow.op.model.constants import ERROR, RESPONSE
from uniflow.op.prompt import Context


class MmModel(AbsModel):
    """Multimodal (MM) Model Class.

    It handles serialization and deserialization of data.
    """

    def _serialize(self, data: List[Context]) -> List[str]:
        """Serialize data.

        Args:
            data (List[Context]): Data to serialize.

        Returns:
            List[str]: Serialized data.
        """
        output = []
        for d in data:
            if not isinstance(d, Context):
                raise ValueError("Input data must be a Context object.")
            prompt_template = copy.deepcopy(self._prompt_template)
            if prompt_template.few_shot_prompt:
                raise ValueError(
                    "Few shot prompt is not supported yet for MultiModal, please set it to None."
                )
            # this is the format for gemini-pro-vision model using Google AI studio
            # https://ai.google.dev/tutorials/python_quickstart#generate_text_from_image_and_text_inputs
            # TODO: this is a temporary solution, we need to have a better way to handle multimodal model
            # for different model providers.
            output.append([prompt_template.instruction, d.context])

        return output

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """
        return {
            RESPONSE: data,
            ERROR: "No errors.",
        }
