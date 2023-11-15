"""Flow Factory Module."""


class FlowFactory:
    """Flow Factory Class."""

    _flows = {}

    @classmethod
    def register(cls, name: str, flow_cls: "Flow") -> None:  # noqa: F821
        """Register flow.

        Args:
            name (str): Flow name.
            flow_cls (Flow): Flow class.
        """
        cls._flows[name] = flow_cls

    @classmethod
    def get(cls, name: str):
        """Get flow.

        Args:
            name (str): Flow name.

        Raises:
            ValueError: If no flow registered under the name.
        """
        flow_cls = cls._flows.get(name)
        if not flow_cls:
            raise ValueError(f"No flow registered under '{name}'")
        return flow_cls
