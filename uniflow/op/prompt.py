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


class PromptTemplate(BaseModel):
    """Type for guided prompt."""

    instruction: str = Field(..., min_length=0)

    few_shot_prompt: conlist(Context, min_length=0) = Field([], min_items=0)

    model_config = ConfigDict(extra="forbid")
