"""Flow config module."""

from dataclasses import dataclass

from uniflow.extract.config import ExtractConfig, ExtractTxtConfig
from uniflow.transform.config import TransformConfig, TransformOpenAIConfig


@dataclass
class PipelineConfig:
    """Pipeline Config Class."""

    extract_config: ExtractConfig = ExtractTxtConfig()
    transform_config: TransformConfig = TransformOpenAIConfig()
