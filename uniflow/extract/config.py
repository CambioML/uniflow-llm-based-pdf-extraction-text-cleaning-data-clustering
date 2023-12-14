"""Extract config module."""

from dataclasses import dataclass


@dataclass
class ExtractConfig:
    """Extract Config Class."""

    flow_name: str = "ExtractTxtFlow"
    num_thread: int = 1


@dataclass
class ExtractTxtConfig(ExtractConfig):
    """Extract Txt Config Class."""

    flow_name: str = "ExtractTxtFlow"
    num_thread: int = 1
