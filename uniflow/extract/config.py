"""Flow config module."""

from dataclasses import dataclass


@dataclass
class Config:
    """Flow Config Class."""

    flow_name: str = "ExtractTxtFlow"
    num_thread: int = 1
