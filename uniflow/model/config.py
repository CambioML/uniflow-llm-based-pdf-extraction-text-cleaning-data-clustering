"""Model Config Module."""


from typing import Dict


class ModelConfig:
    """Model Config Class."""

    model_name: str = "gpt-3.5-turbo-1106"
    num_call: int = 3
    temperature: float = 1.5
    response_format: Dict[str, str] = {"type": "text"}
