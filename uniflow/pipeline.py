"""Queue-Based Pipeline for flow streaming with multiple threads."""

from typing import Any, List, Mapping

from uniflow.flow.client import ExtractClient, TransformClient
from uniflow.flow.config import PipelineConfig
from uniflow.op.prompt import Context


class MultiFlowsPipeline:
    """MultiFlowsPipeline contains an extract and a transform client."""

    def __init__(self, config: PipelineConfig) -> None:
        """MultiFlowsPipeline constructor

        Args:
            config (Dict[str, Any]): Config for the pipeline
        """
        self._config = config
        self._extract_client = ExtractClient(self._config.extract_config)
        self._transform_client = TransformClient(self._config.transform_config)

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Run the pipeline with producer-consumer pattern.

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the pipeline

        Returns:
            List[Mapping[str, Any]]: List of outputs from the pipeline
        """
        output = self._extract_client.run(input_list)
        data = [Context(context=t) for o in output for t in o["output"][0]["text"]]
        return self._transform_client.run(data)
