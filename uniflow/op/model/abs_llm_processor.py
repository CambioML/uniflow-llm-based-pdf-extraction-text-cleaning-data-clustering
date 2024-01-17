"""Abstract class for processors."""
import logging
from typing import Any, Dict, List

from uniflow.op.model.constants import ERROR_LIST, MAX_ATTEMPTS
from uniflow.op.model.model_server import ModelServerFactory
from uniflow.op.prompt import Context, PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AbsLLMProcessor:
    """Abstract class for using LLM for data process."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Model class.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        model_server_cls = ModelServerFactory.get(model_config["model_server"])
        self._model_server = model_server_cls(prompt_template, model_config)
        self._prompt_template = prompt_template
        self._num_samples = 1
        # for Huggingface model
        if "num_return_sequences" in model_config:
            self._num_samples = model_config["num_return_sequences"]
        # for OpenAI model
        elif "num_call" in model_config:
            self._num_samples = model_config["num_call"]

    def _serialize(self, data: List[Context]) -> List[str]:
        """Serialize data.

        Args:
            data (List[Context]): Data to serialize.

        Returns:
            List[str]: Serialized data.
        """
        raise NotImplementedError("Serialize method not implemented.")

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """
        raise NotImplementedError("Deserialize method not implemented.")

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run model.

        Args:
            data (List[Dict[str, Any]]): Data to run.

        Returns:
            List[Dict[str, Any]]: Output data.
        """
        serialized_data = self._serialize(data)
        for i in range(MAX_ATTEMPTS):
            data = self._model_server(serialized_data)
            data = self._deserialize(data)
            if ERROR_LIST not in data:
                break
            logger.info("Attempt %s failed, retrying...", i + 1)

        return data
