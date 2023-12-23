"""Queue-Based Pipeline for flow streaming with multiple threads."""

from queue import Empty, Queue
from threading import Thread
from typing import Any, List, Mapping

from uniflow.flow.client import ExtractClient, TransformClient
from uniflow.flow.config import PipelineConfig
from uniflow.op.prompt_schema import Context


class MultiFlowsPipeline:
    """
    A queue-based pipeline to handle multiple processes for Uniflow,
    including a consumer thread and a producer thread. The Producer/Consumer
    pattern decouples processes that produce and consume data at different
    rates.

    The Producer thread is responsible for putting items into the queue if
    it is not full, while the Consumer thread consumes items if there are any.

    Both threads run indefinitely while checking the status of the queue.
    The Producer thread will stop once it has finished putting all the items
    into the queue. The Consumer thread will stop once it has consumed all
    the items from the queue.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """MultiFlowsPipeline constructor

        Args:
            config (Dict[str, Any]): Config for the pipeline
        """
        self._queue = Queue()
        self._config = config
        self._extract_client = ExtractClient(self._config.extract_config)
        self._transform_client = TransformClient(self._config.transform_config)

    def _producer(self, input_list: List[Mapping[str, Any]]) -> None:
        """Producer function to put items into the queue."""
        for item in input_list:
            extracted_data = self._extract_client.run([item])
            self._queue.put(extracted_data)
        self._queue.put(None)  # Signal for the consumer that production is done

    def _consumer(self, output_list: List[Mapping[str, Any]]) -> None:
        """Consumer function to process items from the queue."""
        while True:
            try:
                data = self._queue.get(timeout=10)  # Adjust timeout as needed
                if data is None:
                    break  # End signal received
                reformat_data = []
                for d in data:
                    # rekey data and reformat to list format
                    rd = [Context(context=t) for t in d["output"][0]["text"]]
                    reformat_data.extend(rd)
                transformed_data = self._transform_client.run(reformat_data)
                output_list.append(transformed_data)
            except Empty:
                # Handle empty queue situation if needed
                pass

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Run the pipeline with producer-consumer pattern.

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the pipeline

        Returns:
            List[Mapping[str, Any]]: List of outputs from the pipeline
        """
        output_list = []
        producer_thread = Thread(target=self._producer, args=(input_list,))
        consumer_thread = Thread(target=self._consumer, args=(output_list,))

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        return output_list
