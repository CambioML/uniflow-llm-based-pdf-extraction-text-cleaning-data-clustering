"""Model inference operation."""
import copy

from typing import Any, Mapping
from uniflow.op.basic.linear_op import LinearOp
import openai


class ModelInfOp(LinearOp):
    """Model inference operation.

    Args:
        nodes (Sequence[Node]): Input nodes.

    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the language model to generate outputs for the prompt.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        qaa_list_encoded = copy.deepcopy(value_dict["qaa_list_encoded"])
        qaa_augmented_raw = []
        for _, batch_inputs_string in enumerate(qaa_list_encoded):
            completion_batch = openai.Completion.create(
                prompt=batch_inputs_string,
                model="text-davinci-003",
                temperature=0.2,
                max_tokens=1000  # The maximum number of tokens to generate in the completion
            )
            results_string = completion_batch['choices'][0]['text']
            qaa_augmented_raw.append(results_string)
        return {"qaa_augmented_raw": qaa_augmented_raw}
