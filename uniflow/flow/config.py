"""Flow config module."""

from dataclasses import dataclass
from uniflow.op.config import ExtractConfig, TransformConfig


@dataclass
class PipelineConfig:
    """Pipeline Config Class."""

    extract_config: ExtractConfig
    transform_config: TransformConfig
