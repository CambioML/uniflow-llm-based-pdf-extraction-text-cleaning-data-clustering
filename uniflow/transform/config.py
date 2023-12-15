"""Transform config module."""

from dataclasses import dataclass, field
from typing import Dict

from uniflow.model.config import HuggingfaceModelConfig, ModelConfig, OpenAIModelConfig
from uniflow.schema import GuidedPrompt


@dataclass
class TransformConfig:
    """Transform Base Config Class."""

    flow_name: str
    model_config: ModelConfig
    num_thread: int = 1
    guided_prompt_template: GuidedPrompt = GuidedPrompt()


@dataclass
class TransformOpenAIConfig(TransformConfig):
    """Transform OpenAI Config Class."""

    flow_name: str = "TransformOpenAIFlow"
    model_config: ModelConfig = OpenAIModelConfig()


@dataclass
class TransformHuggingFaceConfig(TransformConfig):
    """Transform Hugging Face Config Class."""

    flow_name: str = "TransformHuggingFaceFlow"
    model_config: ModelConfig = HuggingfaceModelConfig()


@dataclass
class TransformLMQGConfig(TransformConfig):
    """Transform LMQG Config Class."""

    flow_name: str = "TransformLMQGFlow"
    guided_prompt_template: Dict[str, str] = field(default_factory=lambda: {})
    model_config: ModelConfig = OpenAIModelConfig()


@dataclass
class TransformLinearConfig(TransformConfig):
    """Transform Linear Config Class."""

    flow_name: str = "TransformLinearFlow"
    guided_prompt_template: GuidedPrompt = field(default_factory=lambda: {})
    model_config: ModelConfig = field(default_factory=lambda: {})
