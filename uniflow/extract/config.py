"""Extract config module."""

from dataclasses import dataclass
from typing import Optional

from uniflow.model.config import ModelConfig, NougatModelConfig
from uniflow.schema import GuidedPrompt


@dataclass
class ExtractConfig:
    """Extract Config Class."""

    flow_name: str
    num_thread: int = 1
    model_config: Optional[ModelConfig] = None
    guided_prompt_template: Optional[GuidedPrompt] = None


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
class ExtractIpynbConfig(ExtractConfig):
    """Extract Ipynb Config Class."""

    flow_name: str = "ExtractIpynbFlow"
