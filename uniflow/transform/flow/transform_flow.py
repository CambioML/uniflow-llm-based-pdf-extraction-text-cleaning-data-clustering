"""Transform flow."""

from uniflow.flow import Flow
from uniflow.transform.flow.flow_factory import FlowFactory


class TransformFlow(Flow):
    """Transform Flow Class."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        FlowFactory.register(name=cls.__name__, flow_cls=cls)
