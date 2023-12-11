"""Model Module."""
import copy
import json
import logging
from typing import Any, Dict, List

from uniflow.model.server import ModelServerFactory
from uniflow.schema import Context, GuidedPrompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESPONSE = "response"
ERROR = "error"
ERROR_LIST = "error_list"
ERROR_CONTEXT = "error_context"
MAX_ATTEMPTS = 3


class Model:
    """Model Class."""

    def __init__(
        self,
        model_server: str,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Model class.

        Args:
            model_server (str): Model server name.
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        model_server_cls = ModelServerFactory.get(model_server)
        self._model_server = model_server_cls(model_config)
        if isinstance(guided_prompt_template, GuidedPrompt):
            self._guided_prompt_template = guided_prompt_template.get_prompt()
        else:
            self._guided_prompt_template = guided_prompt_template
        self._prompt_keys = []

    def _serialize(self, data: List[Context]) -> List[str]:
        """Serialize data.

        Args:
            data (List[Dict[str, Any]]): Data to serialize.

        Returns:
            List[str]: Serialized data.
        """
        output = []
        for d in data:
            guided_prompt_template = copy.deepcopy(self._guided_prompt_template)
            if "examples" in guided_prompt_template:
                guided_prompt_template["examples"].append(d.dict())
            else:
                guided_prompt_template = d.dict()

            output_strings = []
            # Iterate over each key-value pair in the dictionary
            for key, value in guided_prompt_template.items():
                if isinstance(value, list):
                    # Special handling for the "examples" list
                    for example in value:
                        for ex_key, ex_value in example.items():
                            output_strings.append(f"{ex_key}: {ex_value}")
                else:
                    output_strings.append(f"{key}: {value}")

            # Join all the strings into one large string, separated by new lines
            output_string = "\n".join(output_strings)
            output.append(output_string)
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


class JsonModel(Model):
    """Json Model Class."""

    def __init__(
        self,
        model_server: str,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Few Shot Model class.

        Args:
            model_server (str): Model server name.
            guided_prompt_template (GuidedPrompt): GuidedPrompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__(model_server, guided_prompt_template, model_config)
        examples = guided_prompt_template.examples
        if not examples:
            raise ValueError(
                "No examples found in guided_prompt_template. Examples are required to use the JSON mode."
            )
        self._json_schema = examples[0].get_custom_schema()

    def _serialize(self, data: List[Context]) -> List[str]:
        """Serialize data.

        Args:
            data (List[Dict[str, Any]]): Data to serialize.

        Returns:
            List[str]: Serialized data.
        """
        guided_prompt_template = copy.deepcopy(self._guided_prompt_template)
        output_schema_guide = " in json"  # f"""Provide the parsed json object that matches the following json_schema (do not deviate at all):
        #     {self._json_schema}
        # """

        guided_prompt_template[
            "instruction"
        ] = f"{guided_prompt_template['instruction']}\n\n{output_schema_guide}"

        input_data = []
        for d in data:
            guided_prompt_template["examples"].append(d.dict())
            input_data.append(guided_prompt_template)

        return [json.dumps(d) for d in input_data]

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """
        error_count = 0
        output_list = []
        error_list = []
        error_context = []

        for d in data:
            try:
                output_list.append(json.loads(d))
            except Exception as e:
                error_count += 1
                error_list.append(str(e))
                error_context.append(d)
                continue

        if error_count == 0:
            return {
                RESPONSE: output_list,
                ERROR: "No errors.",
            }
        return {
            RESPONSE: output_list,
            ERROR: f"Failed to deserialize {error_count} examples",
            ERROR_LIST: error_list,
            ERROR_CONTEXT: error_context,
        }
