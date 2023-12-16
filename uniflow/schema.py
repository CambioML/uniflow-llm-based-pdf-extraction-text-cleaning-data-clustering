"""Schema for uniflow."""

import logging
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, conlist

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

    context: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="allow")

    def get_custom_schema(self) -> Dict[str, Any]:
        """Generate custom schema for the model.

        Args:
            None

        Returns:
            Dict[str, Any]: Custom schema for the model.
        """
        # Extract the known fields
        known_fields = self.model_json_schema()["properties"]

        schema_dict = self.model_dump()

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

    instruction: str = """Generate one question and its corresponding answer based on the last context in the last
    example. Follow the format of the examples below to include context, question, and answer in the response"""

    examples: conlist(Context, min_length=0)

    model_config = ConfigDict(extra="forbid")

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
