"""Model Config Module."""


from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    """Model Config Class."""

    model_name: str = "gpt-3.5-turbo-1106"
    num_call: int = 3
    temperature: float = 1.5
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})
