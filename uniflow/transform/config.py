"""Transform config module."""

from dataclasses import dataclass


@dataclass
class Config:
    """Transform Config Class."""

    flow_name: str = "TransformFlow"
    num_thread: int = 1
