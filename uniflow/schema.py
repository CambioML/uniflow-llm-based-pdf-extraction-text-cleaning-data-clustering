"""Schema for uniflow."""

import logging
from typing import Any, Dict, Union

from pydantic import BaseModel, Extra, Field, conlist

logger = logging.getLogger(__name__)

python_to_json_types = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "None": "null",
}


class Context(BaseModel):
    """Type for prompt context."""

    context: str = Field(..., min_length=10)

    class Config:
        extra = Extra.allow

    def get_custom_schema(self) -> Dict[str, Any]:
        """Generate custom schema for the model.

        Args:
            None

        Returns:
            Dict[str, Any]: Custom schema for the model.
        """
        # Extract the known fields
        known_fields = self.schema()["properties"]

        schema_dict = self.dict()

        # Get keys that are not part of known_fields
        extra_properties = [key for key in schema_dict if key not in known_fields]

        schema = self.model_json_schema()

        # If there are extra properties, add them to the schema
        for prop in extra_properties:
            schema["properties"][prop] = {
                "type": python_to_json_types[type(schema_dict[prop]).__name__],
            }

        return schema["properties"]


class GuidedPrompt(BaseModel):
    """Type for guided prompt."""

    system_prompt: str = ""  # "As a genius expert, your task is to follow the instruction, understand the context and pattern in the examples."

    instruction: str = "Generate one question and its corresponding answer based on the last context in the last example. Following the format of the examples below to include context, question, and answer in the response"

    examples: conlist(Union[Context], min_length=0)

    class Config:
        extra = Extra.forbid

    def __init__(self, **data):
        """Initialize GuidedPrompt class.

        Args:
            data (Dict[str, Any]): Data to initialize.
        """
        default_prompt_qa_1 = Context(
            context="The quick brown fox jumps over the lazy black dog.",
            question="What is the color of the fox?",
            answer="brown.",
        )
        default_prompt_qa_2 = Context(
            context="The quick brown fox jumps over the lazy black dog.",
            question="What is the color of the dog?",
            answer="black.",
        )
        data.setdefault("examples", [default_prompt_qa_1, default_prompt_qa_2])
        super().__init__(**data)

    def get_prompt(self):
        """Create prompt for the model.

        Args:
            None

        Returns:
            Dict[str, Any]: Prompt for the model.
        """
        return {
            "instruction": self.system_prompt + "\n" + self.instruction,
            "examples": [example.dict() for example in self.examples],
        }
