"""Model Op Module."""
import copy
import json
import logging
from typing import Any, Dict, List, Sequence

from uniflow.node import Node
from uniflow.op.model.model_server import ModelServerFactory
from uniflow.op.op import Op
from uniflow.op.prompt_schema import Context, GuidedPrompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESPONSE = "response"
ERROR = "error"
ERROR_LIST = "error_list"
ERROR_CONTEXT = "error_context"
MAX_ATTEMPTS = 3


class LLMDataProcessor:
    """
    Data processor for processing data with a LLM model server.
    It handles serialization and deserialization of data,
    manages interactions with the LLM model server, and applies
    a guided prompt template to the data.
    """

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize Model class.

        Args:
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        model_server_cls = ModelServerFactory.get(model_config["model_server"])
        self._model_server = model_server_cls(model_config)
        self._guided_prompt_template = guided_prompt_template
        self._prompt_keys = []

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
            output_strings = []
            guided_prompt_template = copy.deepcopy(self._guided_prompt_template)
            if (
                not guided_prompt_template.instruction
                and not guided_prompt_template.examples
            ):
                for key, value in d.model_dump().items():
                    output_strings.append(f"{key}: {value}")
            else:
                guided_prompt_template.examples.append(d)
                output_strings.append(
                    f"instruction: {guided_prompt_template.instruction}"
                )
                for example in guided_prompt_template.examples:
                    for ex_key, ex_value in example.model_dump().items():
                        output_strings.append(f"{ex_key}: {ex_value}")

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


class ModelOp(Op):
    """Model Op Class."""

    def __init__(self, name: str, model: LLMDataProcessor) -> None:
        """Model Op Constructor.

        Args:
            name (str): Name of the op.
            model (Model): Model to run.
        """
        super().__init__(name)
        self._model = model

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Op.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            value_dict = self._model.run(value_dict)
            output_nodes.append(
                Node(name=self.unique_name(), value_dict=value_dict, prev_nodes=[node])
            )
        return output_nodes
