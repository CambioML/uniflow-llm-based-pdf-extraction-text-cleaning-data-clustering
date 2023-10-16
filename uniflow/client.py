"""Uniflow Client"""

from typing import Any, List, Mapping
from uniflow.server import Server
from uniflow.constants import flows_dict


class Client:
    """Uniflow Client"""

    def __init__(self, flow_name: str) -> None:
        """Client constructor

        Args:
            flow_name (str): Name of the flow to run

        """
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
