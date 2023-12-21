"""Model Module."""
import copy
import json
from typing import Any, Dict, List

from uniflow.op.prompt_schema import Context, GuidedPrompt
from uniflow.op.transform.model_op import LLMDataProcessor


class JsonFormattedDataProcessor(LLMDataProcessor):
    """
    Extends the LLMDataProcessor Class to ensure the response is in json.
    """

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Json Model class.

        Args:
            guided_prompt_template (GuidedPrompt): GuidedPrompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__(guided_prompt_template, model_config)
        examples = guided_prompt_template.examples
        if not examples:
            raise ValueError(
                "No examples found in guided_prompt_template. Examples are required to use the JSON mode."
            )
        self._json_schema = examples[0].get_custom_schema()

    def _serialize(self, data: List[Context]) -> List[str]:
        """Serialize data.

        Args:
            data (List[Context]): Data to serialize.

        Returns:
            List[str]: Serialized data.
        """
        for d in data:
            if not isinstance(d, Context):
                raise ValueError("Input data must be a Context object.")
            guided_prompt_template = copy.deepcopy(self._guided_prompt_template)
            output_schema_guide = "Ensure the response is in json."
            # f"""Provide the parsed json object
            # that matches the following json_schema (do not deviate at all):
            #     {self._json_schema}
            # """

            guided_prompt_template.instruction = (
                f"{guided_prompt_template.instruction}\n\n{output_schema_guide}"
            )

            input_data = []
            guided_prompt_template.examples.append(d)
            input_data.append(guided_prompt_template.model_dump())
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
            except json.JSONDecodeError as e:
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


class LLMDataPreprocessor(LLMDataProcessor):
    """Preprocess Model Class."""

    def __init__(
        self,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Preprocess Model class.

        Args:
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__(guided_prompt_template={}, model_config=model_config)

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
