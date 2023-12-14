"""Flow __init__ module."""
# this register all possible flow into FlowFactory through
# FlowFactory.register(cls.__name__, cls) in Flow __init_subclass__

from uniflow.transform.flow.transform_flow import TransformFlow  # noqa: F401, F403

__all__ = ["TransformFlow"]
