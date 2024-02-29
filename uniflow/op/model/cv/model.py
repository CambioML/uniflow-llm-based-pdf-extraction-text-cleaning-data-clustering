"""Computer vision (CV) model class."""

import logging
from typing import Any, Dict, List

from uniflow.op.model.abs_model import AbsModel
from uniflow.op.model.constants import ERROR, RESPONSE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CvModel(AbsModel):
    """Computer Vision (CV) Model Class.

    It handles serialization and deserialization of data.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Preprocess Model class.

        Args:
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__(prompt_template=None, model_config=model_config)

    def _serialize(self, data: List[Dict[str, Any]]) -> List[str]:
        """Serialize data.

        Args:
            data (List[Dict[str, Any]]): Data to serialize.

        Returns:
            List[str]: Serialized data.
        """
        output = []
        for value in data.values():
            output.append(value)
        return output

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """
        output_list = []
        error_count = 0

        for d in data:
            try:
                output_list.append(d)
            except Exception:
                error_count += 1
                continue
        return {
            RESPONSE: output_list,
            ERROR: f"Failed to deserialize {error_count} examples",
        }
