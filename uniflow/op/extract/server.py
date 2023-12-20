"""Uniflow Extract Server"""

from concurrent import futures
from queue import Queue
from typing import Any, Dict, List, Mapping, Tuple

from tqdm.auto import tqdm

from uniflow.constants import EXTRACT
from uniflow.op.config import ExtractConfig
from uniflow.flow.flow_factory import FlowFactory
from uniflow.op.op import OpScope


class Server:
    """Uniflow Extract Server"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Server constructor

        Args:
            config (Flow): Flow class to run
            num_thread (int, optional): Number of threads to run the flow. Defaults to 1.
        """
        # convert from dict to config for type checking
        self._config = ExtractConfig(**config)

        self._flow_cls = FlowFactory.get(self._config.flow_name, flow_type=EXTRACT)
        self._num_thread = self._config.num_thread
        self._flow_queue = Queue(self._num_thread)
        kwargs = {}
        if self._config.model_config:
            kwargs["model_config"] = self._config.model_config
        for i in range(self._num_thread):
            with OpScope(name="thread_" + str(i)):
                self._flow_queue.put(self._flow_cls(**kwargs))

    def _run_flow(
        self, input_list: Mapping[str, Any], index: int
    ) -> Tuple[int, Mapping[str, Any]]:
        """Run the flow

        Args:
            input_list (Mapping[str, Any]): Input to the flow
            index (int): Index of the input

        Returns:
            Tuple[int, Mapping[str, Any]]: Index of the output, Output from the flow
        """
        f = self._flow_queue.get(timeout=0)
        ###########################################
        # this is very import to prevent deadlock #
        ###########################################
        try:
            output = f(input_list)
        except Exception as e:
            output = {"error": str(e)}
        self._flow_queue.put(f)
        return (index, output)

    def _run_flow_wrapper(
        self, input_list: Mapping[str, Any], i: int
    ) -> Tuple[int, Mapping[str, Any]]:
        """Wrapper for _run_flow

        Args:
            input_list (Mapping[str, Any]): Input to the flow
            i (int): Index of the input

        Returns:
            Tuple[int, Mapping[str, Any]]: Index of the output, Output from the flow
        """
        return self._run_flow(input_list, i)

    def run(self, input_list: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Run the flow

        Args:
            input_list (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        with futures.ThreadPoolExecutor(max_workers=self._num_thread) as executor:
            output_futures = {
                executor.submit(self._run_flow_wrapper, input_data, i): i
                for i, input_data in enumerate(input_list)
            }
            results = [None] * len(input_list)

            for future in tqdm(
                futures.as_completed(output_futures), total=len(input_list)
            ):
                index = output_futures[future]
                results[index] = future.result()[1]
        return results

    def async_run(self):
        """Run the flow asynchronously"""
        # TODO: Implement async server
        print("Server running async")
