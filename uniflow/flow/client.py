"""All Uniflow Clients"""

from dataclasses import asdict, is_dataclass
from typing import Any, List, Mapping

import tiktoken

from uniflow.flow.config import ExtractConfig, RaterConfig
from uniflow.flow.server import ExtractServer, RaterServer, TransformServer


def dataclass_to_dict(dataclass_instance):
    """Convert dataclass to dict"""
    if is_dataclass(dataclass_instance):
        return {k: dataclass_to_dict(v) for k, v in asdict(dataclass_instance).items()}
    if isinstance(dataclass_instance, (list, tuple)):
        return [dataclass_to_dict(v) for v in dataclass_instance]
    return dataclass_instance


class ExtractClient:
    """Uniflow Extract Client"""

    def __init__(self, config: ExtractConfig) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config
        self._server = ExtractServer(dataclass_to_dict(self._config))

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


class TransformClient:
    """Uniflow Transform Client"""

    def __init__(self, name: str) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._server = TransformServer(name)
        self._encoder = tiktoken.encoding_for_model("gpt-3.5")
        self._token_size_limit = 4096  # Define the token size limitation

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """
        Process and possibly split input contexts before running them through the server.

        This method checks if auto-splitting of long text is enabled in the configuration. If so, it evaluates each input context.
        If a context exceeds a certain size, it is split into smaller chunks to meet the token size limitations.
        Each chunk is then processed separately. If auto-splitting is not enabled, or if the context is within the acceptable size limit,
        the input is processed as is.

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        # Pass the processed input to the server for further processing
        output = self._server.run(input_list)
        return output

    def async_run(self) -> None:
        """Run the flow asynchronously"""
        # TODO: Implement async client
        print("Client running async")


class RaterClient:
    """Uniflow Rater Client"""

    def __init__(self, config: RaterConfig) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config
        self._server = RaterServer(asdict(self._config))

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
