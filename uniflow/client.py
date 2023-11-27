"""Uniflow Client"""

from dataclasses import asdict
from typing import Any, List, Mapping

from uniflow.config import Config
from uniflow.server import Server


class Client:
    """Uniflow Client"""

    def __init__(self, config: Config) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config

    def run(self, input: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """
        Run the flow

        Args:
            input (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        # convert config to dict for future remote calls
        server = Server(asdict(self._config))
        output = server.run(input)
        return output

    def async_run(self) -> None:
        # TODO: Implement async client
        print("Client running async")
