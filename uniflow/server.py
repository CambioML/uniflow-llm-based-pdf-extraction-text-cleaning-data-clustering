"""Uniflow Server"""

import traceback
from concurrent import futures
from queue import Queue
from typing import Any, Dict, List, Mapping, Tuple

from tqdm.auto import tqdm

from uniflow.config import Config
from uniflow.flow.flow_factory import FlowFactory
from uniflow.op.op import OpScope


class Server:
    """Uniflow Server"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Server constructor

        Args:
            config (Flow): Flow class to run
            num_thread (int, optional): Number of threads to run the flow. Defaults to 1.
        """
        # convert from dict to config for type checking
        self._config = Config(**config)

        self._flow_cls = FlowFactory.get(self._config.flow_name)
        self._num_thread = self._config.num_thread
        self._flow_queue = Queue(self._num_thread)
        for i in range(self._num_thread):
            with OpScope(name="thread_" + str(i)):
                self._flow_queue.put(
                    self._flow_cls(
                        self._config.model_config["model_server"],
                        self._config.guided_prompt_template,
                        self._config.model_config,
                    )
                )

    def _run_flow(
        self, input_list: Mapping[str, Any], index: int
    ) -> Tuple[int, Mapping[str, Any]]:
        """Run the flow

        Args:
            input_list (Mapping[str, Any]): Input to the flow
            index (int): Index of the input_list

        Returns:
            Tuple[int, Mapping[str, Any]]: Index of the output, Output from the flow
        """
        f = self._flow_queue.get(timeout=0)
        ###########################################
        # this is very import to prevent deadlock #
        ###########################################
        # TODO: update to fail immediately without continue through config.
        try:
            output = f(input_list)
        except Exception as e:
            # Capture the full stack trace
            error_traceback = traceback.format_exc()
            output = {"error": str(e), "traceback": error_traceback}
        self._flow_queue.put(f)
        return (index, output)

    def _run_flow_wrapper(
        self, input_list: Mapping[str, Any], i: int
    ) -> Tuple[int, Mapping[str, Any]]:
        """Wrapper for _run_flow

        Args:
            input_list (Mapping[str, Any]): Input to the flow
            i (int): Index of the input_list

        Returns:
            Tuple[int, Mapping[str, Any]]: Index of the output, Output from the flow
        """
        return self._run_flow(input_list, i)

    def _divide_data_into_batches(
        self, input_list: List[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        """Divide the list into batches

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of batches
        """
        # currently only HuggingFace model support batch.
        # this will require some refactoring to support other models.
        batch_size = self._config.model_config.get(  # pylint: disable=no-member
            "batch_size", 1
        )
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if not input_list:  # Check if the list is empty
            return []

        # Main logic to divide the list into batches
        batched_list = []
        for i in range(0, len(input_list), batch_size):
            batched_list.append(input_list[i : i + batch_size])  # noqa: E203
        return batched_list

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Run the flow

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        batch_data = self._divide_data_into_batches(input_list)
        with futures.ThreadPoolExecutor(max_workers=self._num_thread) as executor:
            output_futures = {
                executor.submit(self._run_flow_wrapper, input_data, i): i
                for i, input_data in enumerate(batch_data)
            }
            # use batch_data size to initialize results
            results = [None] * len(batch_data)

            for future in tqdm(
                futures.as_completed(output_futures), total=len(batch_data)
            ):
                index = output_futures[future]
                results[index] = future.result()[1]
        return results

    def async_run(self):
        # TODO: Implement async server
        print("Server running async")
