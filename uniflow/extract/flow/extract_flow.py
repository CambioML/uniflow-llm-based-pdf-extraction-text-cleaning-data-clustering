"""Extract Flow class."""
from uniflow.extract.flow.flow_factory import FlowFactory
from uniflow.flow import Flow


class ExtractFlow(Flow):
    """Extract Flow class."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        FlowFactory.register(name=cls.__name__, flow_cls=cls)
