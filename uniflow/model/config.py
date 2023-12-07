"""Model Config Module."""


from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    """Model Config Class."""

    model_name: str = "gpt-3.5-turbo-1106"


@dataclass
class OpenAIModelConfig(ModelConfig):
    """OpenAI Model Config Class."""

    num_call: int = 1
    temperature: float = 1.5
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})


@dataclass
class OpenAIJsonModelConfig(ModelConfig):
    """OpenAI Model Config Class."""

    num_call: int = 1
    temperature: float = 1.5
    response_format: Dict[str, str] = field(
        default_factory=lambda: {"type": "json_object"}
    )


@dataclass
class HuggingfaceModelConfig(ModelConfig):
    """Huggingface Model Config Class."""

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    batch_size: int = 1


@dataclass
class LMQGModelConfig(ModelConfig):
    """LMQG Model Config Class."""

    model_name: str = "lmqg/t5-base-squad-qg-ae"
    batch_size: int = 1
