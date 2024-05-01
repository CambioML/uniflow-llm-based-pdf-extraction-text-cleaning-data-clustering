"""All Uniflow Clients"""

from dataclasses import asdict, is_dataclass
from typing import Any, List, Mapping

import tiktoken

from uniflow.flow.config import ExtractConfig, RaterConfig, TransformConfig
from uniflow.flow.server import ExtractServer, RaterServer, TransformServer
from uniflow.node import Node
from uniflow.op.extract.split.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)
from uniflow.op.prompt import Context


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

    def __init__(self, config: TransformConfig) -> None:
        """Client constructor

        Args:
            config (Config): Config for the flow

        """
        self._config = config
        self._server = TransformServer(asdict(self._config))
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

        processed_input = []

        # Check if auto-splitting of long text is enabled
        if self._config.auto_split_long_text:
            # Iterate over each input context
            for input_item in input_list:
                # Calculate the length of the context in characters
                context_length = len(self._encoder.encode(input_item.context))

                # Compare context length with the token size limit
                if context_length > self._token_size_limit:
                    # Log the need for splitting the context due to size exceeding the limit
                    # print("The current Context object needs splitting because it exceeds the token limitation.")

                    # Set an adjusted chunk size for splitting
                    adjusted_chunk_size = (
                        self._token_size_limit
                    )  # This size may need adjustment based on tokenization characteristics

                    # Initialize the splitter with the calculated chunk size and overlap
                    splitter = RecursiveCharacterSplitter(
                        name="text_splitter",
                        chunk_size=adjusted_chunk_size,
                        chunk_overlap_size=50,
                        splitting_mode="token",  # This is the key update to use token-based splitting
                    )

                    # Create a node from the current context for splitting
                    nodes = [
                        Node(name="input_node", value_dict={"text": input_item.context})
                    ]

                    # Split the context into smaller chunks
                    split_nodes = splitter(nodes)

                    # Process each split chunk
                    for node in split_nodes:
                        chunk_text = node.value_dict["text"]
                        for c_text in chunk_text:
                            # Create a new Context object for each chunk and add it to the processed list
                            chunk_context = Context(context=c_text)
                            processed_input.append(chunk_context)
                else:
                    # If the context is within the acceptable size limit, add it directly to the processed list
                    processed_input.append(input_item)
        else:
            # If auto-splitting is not enabled, use the input list as is
            processed_input = input_list

        # Pass the processed input to the server for further processing
        output = self._server.run(processed_input)
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
