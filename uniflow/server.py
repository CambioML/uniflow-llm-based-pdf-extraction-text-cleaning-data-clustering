"""Uniflow Server"""

from concurrent import futures
from typing import Any, List, Mapping, Tuple
from queue import Queue
from uniflow.flow.flow import Flow
from uniflow.op.op import OpScope


class Server:
    """Uniflow Server"""

    def __init__(self, flow_cls: Flow, num_thread: int = 2) -> None:
        """Server constructor

        Args:
            flow_cls (Flow): Flow class to run
            num_thread (int, optional): Number of threads to run the flow. Defaults to 2.
        """
        self._flow_cls = flow_cls
        self._num_thread = num_thread
        self._flow_queue = Queue(self._num_thread)
        for i in range(self._num_thread):
            with OpScope(name="thread_" + str(i)):
                self._flow_queue.put(self._flow_cls())

    def _run_flow(
        self, input: Mapping[str, Any], index: int
    ) -> Tuple[int, Mapping[str, Any]]:
        """Run the flow

        Args:
            input (Mapping[str, Any]): Input to the flow
            index (int): Index of the input

        Returns:
            Tuple[int, Mapping[str, Any]]: Index of the output, Output from the flow
        """
        f = self._flow_queue.get()
        output = f(input)
        self._flow_queue.put(f)
        return (index, output)

    def _run_flow_wrapper(
        self, input: Mapping[str, Any], i: int
    ) -> Tuple[int, Mapping[str, Any]]:
        """Wrapper for _run_flow

        Args:
            input (Mapping[str, Any]): Input to the flow
            i (int): Index of the input

        Returns:
            Tuple[int, Mapping[str, Any]]: Index of the output, Output from the flow
        """
        return self._run_flow(input, i)

    def run(self, input: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Run the flow

        Args:
            input (List[Mapping[str, Any]]): List of inputs to the flow

        Returns:
            List[Mapping[str, Any]]: List of outputs from the flow
        """
        with futures.ThreadPoolExecutor(max_workers=self._num_thread) as executor:
            output_futures = {
                executor.submit(self._run_flow_wrapper, input_data, i): i
                for i, input_data in enumerate(input)
            }
            results = [None] * len(input)
            for future in futures.as_completed(output_futures):
                index = output_futures[future]
                results[index] = future.result()[1]
        return results

    def async_run(self):
        # TODO: Implement async server
        print("Server running async")
