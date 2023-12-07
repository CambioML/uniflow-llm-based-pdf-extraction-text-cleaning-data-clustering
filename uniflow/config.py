"""Flow config module."""

from dataclasses import dataclass, field
from typing import Dict

from uniflow.model.config import (
    HuggingfaceModelConfig,
    LMQGModelConfig,
    ModelConfig,
    OpenAIJsonModelConfig,
    OpenAIModelConfig,
)
from uniflow.schema import FewShotTemplate


@dataclass
class Config:
    """Flow Config Class."""

    flow_name: str = "ModelFlow"
    few_shot_template: Dict[str, str] = field(default_factory=lambda: {})
    num_thread: int = 1
    model_config: ModelConfig = ModelConfig()


@dataclass
class OpenAIConfig:
    """Flow Config Class."""

    flow_name: str = "ModelFlow"
    few_shot_template: Dict[str, str] = field(default_factory=lambda: {})
    num_thread: int = 1
    model_config: ModelConfig = OpenAIModelConfig()


@dataclass
class OpenAIJsonConfig:
    """Flow Config Class."""

    flow_name: str = "OpenAIJsonModelFlow"
    few_shot_template: Dict[str, str] = field(default_factory=lambda: {})
    num_thread: int = 1
    model_config: ModelConfig = OpenAIJsonModelConfig()


@dataclass
class OpenAIFewShotConfig:
    """Flow Config Class."""

    flow_name: str = "FewShotModelFlow"
    few_shot_template: Dict[str, str] = field(
        default_factory=lambda: FewShotTemplate().dict()
    )
    num_thread: int = 1
    model_config: ModelConfig = OpenAIModelConfig()


@dataclass
class HuggingfaceConfig:
    """Huggingface Config Class."""

    flow_name: str = "HuggingFaceModelFlow"
    few_shot_template: Dict[str, str] = field(default_factory=lambda: {})
    num_thread: int = 1
    model_config: ModelConfig = HuggingfaceModelConfig()


@dataclass
class LMQGConfig:
    """LMQG Config Class."""

    flow_name: str = "LMQGModelFlow"
    few_shot_template: Dict[str, str] = field(default_factory=lambda: {})
    num_thread: int = 1
    model_config: ModelConfig = LMQGModelConfig()
