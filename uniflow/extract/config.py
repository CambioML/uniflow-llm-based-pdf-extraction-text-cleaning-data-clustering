"""Extract config module."""

from dataclasses import dataclass


@dataclass
class Config:
    """Extract Config Class."""

    flow_name: str = "ExtractTxtFlow"
    num_thread: int = 1
