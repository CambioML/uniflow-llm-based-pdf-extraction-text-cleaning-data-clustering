"""Model Op Module."""
import copy
import logging
from typing import Any, Dict, List, Sequence

from uniflow.node.node import Node
from uniflow.op.op import Op
from uniflow.op.prompt_schema import Context, GuidedPrompt
from uniflow.op.transform.model_server import ModelServerFactory

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
