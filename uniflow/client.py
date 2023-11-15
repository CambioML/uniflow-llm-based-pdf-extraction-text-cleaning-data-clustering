"""Uniflow Client"""

from typing import Any, List, Mapping

from uniflow.constants import flows_dict
from uniflow.flow.flow_factory import FlowFactory
from uniflow.server import Server


class Client:
    """Uniflow Client"""

    def __init__(self, flow_name: str) -> None:
        """Client constructor

        Args:
            flow_name (str): Name of the flow to run

        """
        try:
            self._flow_cls = FlowFactory.get(flow_name)
        except Exception:
            # TODO (Jojo): remove this after all flow is registered
            print(
                f"Flow {flow_name} not found in flow factory and use old flow instead"
            )
            self._flow_cls = flows_dict[flow_name]

    def run(self, input: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """
        Run the flow

        Args:
            input (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        server = Server(self._flow_cls)
        output = server.run(input)
        return output

    def async_run(self) -> None:
        # TODO: Implement async client
        print("Client running async")
