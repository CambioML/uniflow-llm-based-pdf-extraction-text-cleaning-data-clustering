"""Flow Factory Module."""
from typing import Dict, List

from uniflow.constants import EXTRACT, RATER, TRANSFORM


class FlowFactory:
    """Flow Factory Class."""

    _flows = {EXTRACT: {}, TRANSFORM: {}, RATER: {}}

    @classmethod
    def register(cls, name: str, flow_cls: "Flow") -> None:  # noqa: F821
        """Register flow.

        Args:
            name (str): Flow name.
            flow_cls (Flow): Flow class.
        """
        if hasattr(flow_cls, "TAG") and flow_cls.TAG in cls._flows:
            cls._flows[flow_cls.TAG][name] = flow_cls

    @classmethod
    def get(cls, name: str, flow_type: str) -> "Flow":  # noqa: F821
        """Get flow.

        Args:
            name (str): Flow name.

        Returns:
            Flow: Flow.

        Raises:
            ValueError: If no flow registered under the name.
        """
        flow_cls = cls._flows.get(flow_type).get(name)
        if not flow_cls:
            raise ValueError(f"No flow registered under '{name}'")
        return flow_cls

    @classmethod
    def list(cls) -> Dict[str, List[str]]:
        """List all registered flows.

        Returns:
            List[str]: List of registered flow names.
        """
        return {
            subclass: list(subclass_dict.keys())
            for subclass, subclass_dict in cls._flows.items()
        }
