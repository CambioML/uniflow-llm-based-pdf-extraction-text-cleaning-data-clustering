"""All Uniflow Clients"""

from dataclasses import asdict
from typing import Any, List, Mapping
from uniflow.op.prompt import Context
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

    def run(self, input_list: List[Context]) -> List[Mapping[str, Any]]:
        processed_input = []

        # Check if auto-splitting of long text is enabled
        if self._config.auto_split_long_text:
            # Define the token size limitation
            token_size_limit = 10000

            # Iterate over each input context
            for input_item in input_list:
                # Directly use the character count as a proxy for size, though not perfectly accurate
                context_length = len(input_item.context)

                # If the context length is larger than a conservative estimate for the token size limit, split it
                if context_length > token_size_limit:
                    # Initialize the splitter with an adjusted chunk size based on the need to stay under the token limit
                    # This size should be adjusted based on empirical data about the average token size for the text in question
                    print("The current Context object needs splitting because it exceeds the token limitation.")
                    adjusted_chunk_size = token_size_limit  # Adjust based on empirical tokenization data
                    splitter = RecursiveCharacterSplitter(name='text_splitter', chunk_size=adjusted_chunk_size, chunk_overlap_size=50)
                    nodes = [Node(name='input_node', value_dict={'text': input_item.context})]
                    split_nodes = splitter(nodes)

                    # Convert each chunk into a Context object and add to processed input
                    # print(len(split_nodes))
                    for node in split_nodes:
                        chunk_text = node.value_dict['text']
                        # print(chunk_text)
                        # print(len(chunk_text[0]))
                        # print(chunk_text[0])
                        for c_text in chunk_text:
                            print(len(c_text))
                            chunk_context = Context(context=c_text)
                            processed_input.append(chunk_context)
                else:
                    # If the context length is within the limit, use it as is
                    processed_input.append(input_item)
        else:
            # If auto-splitting is not enabled, use the original input list
            processed_input = input_list

        # Pass the processed input to the server and return the output
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
