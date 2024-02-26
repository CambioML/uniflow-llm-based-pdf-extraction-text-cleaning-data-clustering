"""All Uniflow Clients"""

from dataclasses import asdict
from typing import Any, List, Mapping
from uniflow.op.extract.split.recursive_character_splitter import RecursiveCharacterSplitter
from uniflow.node import Node

from uniflow.flow.config import ExtractConfig, RaterConfig, TransformConfig
from uniflow.flow.server import ExtractServer, RaterServer, TransformServer


class ExtractClient:
    """Uniflow Extract Client"""

    def __init__(self, config: ExtractConfig) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config
        self._server = ExtractServer(asdict(self._config))

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        # Initialize the splitter with the desired chunk size and overlap size
        splitter = RecursiveCharacterSplitter(name='text_splitter', chunk_size=4000, chunk_overlap_size=50)
        
        output = []

        # Iterate over each input context
        for input_item in input_list:
            # Use the splitter to split the text into chunks if necessary
            text = input_item['context']
            nodes = [Node(value_dict={'text': text})]
            split_nodes = splitter(nodes)

            # Process each chunk as a separate context
            for node in split_nodes:
                chunk_text = node.value_dict['text']
                chunk_output = self._server.run([{'context': chunk_text}])
                output.append(chunk_output)

        # Assuming the outputs are already in the correct format, just return them
        return output

    def async_run(self) -> None:
        """Run the flow asynchronously"""
        # TODO: Implement async client
        print("Client running async")


class TransformClient:
    """Uniflow Transform Client"""

    def __init__(self, config: TransformConfig) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config
        self._server = TransformServer(asdict(self._config))

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
