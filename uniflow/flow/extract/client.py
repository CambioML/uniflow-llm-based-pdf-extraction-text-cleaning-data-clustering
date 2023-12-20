"""Uniflow Extract Client"""

from dataclasses import asdict
from typing import Any, List, Mapping

from uniflow.op.config import ExtractConfig
from uniflow.flow.extract.server import Server


class Client:
    """Uniflow Extract Client"""

    def __init__(self, config: ExtractConfig) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config
        self._server = Server(asdict(self._config))

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """
        Run the flow

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        # convert config to dict for future remote calls
        output = self._server.run(input_list)
        return output

    def async_run(self) -> None:
        """Run the flow asynchronously"""
        # TODO: Implement async client
        print("Client running async")
