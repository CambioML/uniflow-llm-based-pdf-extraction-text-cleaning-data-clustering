"""Flow config module."""

from dataclasses import dataclass

from uniflow.extract.config import ExtractConfig
from uniflow.transform.config import TransformConfig


@dataclass
class PipelineConfig:
    """Pipeline Config Class."""

    extract_config: ExtractConfig
    transform_config: TransformConfig
