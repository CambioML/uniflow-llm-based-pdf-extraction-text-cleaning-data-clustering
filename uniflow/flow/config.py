"""Extract, Transform and Pipeline config module."""

from dataclasses import dataclass, field
from typing import Optional

from uniflow.op.model.model_config import (
    HuggingfaceModelConfig,
    ModelConfig,
    NougatModelConfig,
    OpenAIModelConfig,
)
from uniflow.op.prompt_schema import GuidedPrompt

###########################################################
#                   All Extract Configs                   #
###########################################################


@dataclass
class ExtractConfig:
    """Extract Config Class."""

    flow_name: str
    num_thread: int = 1
    model_config: Optional[ModelConfig] = None


@dataclass
class ExtractTxtConfig(ExtractConfig):
    """Extract Txt Config Class."""

    flow_name: str = "ExtractTxtFlow"


@dataclass
class ExtractPDFConfig(ExtractConfig):
    """Nougat Config Class."""

    flow_name: str = "ExtractPDFFlow"
    model_config: ModelConfig = NougatModelConfig()


@dataclass
class ExtractMarkdownConfig(ExtractConfig):
    """Extract Markdown Config Class."""

    flow_name: str = "ExtractMarkdownFlow"


@dataclass
class ExtractIpynbConfig(ExtractConfig):
    """Extract ipynb Config Class."""

    flow_name: str = "ExtractIpynbFlow"


###########################################################
#                   All Transform Configs                 #
###########################################################


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
    guided_prompt_template: GuidedPrompt = GuidedPrompt(instruction="", examples=[])
    model_config: ModelConfig = OpenAIModelConfig()


@dataclass
class TransformCopyConfig(TransformConfig):
    """Transform Linear Config Class."""

    flow_name: str = "TransformCopyFlow"
    guided_prompt_template: GuidedPrompt = GuidedPrompt(instruction="", examples=[])
    model_config: ModelConfig = field(default_factory=lambda: {})


###########################################################
#                    Pipeline Config                      #
###########################################################
@dataclass
class PipelineConfig:
    """MultiFlowsPipeline Config Class."""

    extract_config: ExtractConfig
    transform_config: TransformConfig
