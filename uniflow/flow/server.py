"""All Servers including Model Server, Extract Server and Transform Server."""

import re
import traceback
from concurrent import futures
from functools import partial
from queue import Queue
from tqdm.auto import tqdm
from typing import Any, Dict, List, Mapping, Tuple

from uniflow.constants import EXTRACT, TRANSFORM
from uniflow.op.config import ExtractConfig, TransformConfig
from uniflow.flow.flow_factory import FlowFactory
from uniflow.op.op import OpScope
from uniflow.op.config import (
    HuggingfaceModelConfig,
    LMQGModelConfig,
    NougatModelConfig,
    OpenAIModelConfig,
)


###############################################################################
#                                Extract Server                               #
###############################################################################

class ExtractServer:
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


###############################################################################
#                             Transform Server                                #
###############################################################################

class TransformServer:
    """Uniflow Transform Server"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Server constructor

        Args:
            config (Flow): Flow class to run
            num_thread (int, optional): Number of threads to run the flow. Defaults to 1.
        """
        # convert from dict to config for type checking
        self._config = TransformConfig(**config)

        self._flow_cls = FlowFactory.get(self._config.flow_name, flow_type=TRANSFORM)
        self._num_thread = self._config.num_thread
        self._flow_queue = Queue(self._num_thread)
        for i in range(self._num_thread):
            with OpScope(name="thread_" + str(i)):
                self._flow_queue.put(
                    self._flow_cls(
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
        """Run the flow asynchronously"""
        # TODO: Implement async server
        print("Server running async")


###############################################################################
#                             All Model Servers                               #
###############################################################################

class ModelServerFactory:
    """Model Server Factory."""

    _servers = {}

    @classmethod
    def register(cls, name: str, server_cls: "AbsModelServer") -> None:
        """Register model server.

        Args:
            name (str): Model server name.
            server_cls (AbsModelServer): Model server class.
        """
        cls._servers[name] = server_cls

    @classmethod
    def get(cls, name: str) -> "AbsModelServer":
        """Get model server.

        Args:
            name (str): Model server name.

        Returns:
            AbsModelServer: Model server.

        Raises:
            ValueError: If no model server registered under the name.
        """
        server_cls = cls._servers.get(name)
        if not server_cls:
            raise ValueError(f"No model server registered under '{name}'")
        return server_cls

    @classmethod
    def list(cls):
        """List all registered model servers.

        Returns:
            List[str]: List of registered model server names.
        """
        return list(cls._servers.keys())


class AbsModelServer:
    """Abstract Model Server Class."""

    def __init_subclass__(cls):
        """Register model server.

        This method is called when subclass is created.
        """
        super().__init_subclass__()
        ModelServerFactory.register(cls.__name__, cls)

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize AbsModelServer class.

        Args:
            model_config (Dict[str, Any]): Model config.
        """
        self._model_config = model_config

    def _preprocess(self, data: str) -> str:
        """Preprocess data.

        Args:
            data (str): Data to preprocess.

        Returns:
            str: Preprocessed data.
        """
        raise NotImplementedError

    def __call__(self, data: str) -> str:
        """Run model.

        Args:
            data (str): Data to run.

        Returns:
            str: Output data.
        """
        raise NotImplementedError

    def _postprocess(self, data: str) -> List[str]:
        """Postprocess data.

        Args:
            data (str): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        raise NotImplementedError


class OpenAIModelServer(AbsModelServer):
    """OpenAI Model Server Class."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        # import in class level to avoid installing openai package
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        super().__init__(model_config)
        self._model_config = OpenAIModelConfig(**self._model_config)
        self._client = OpenAI()

    def _preprocess(self, data: List[str]) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (str): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return [c.message.content for d in data for c in d.choices]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        OpenAI completions API does not support batch inference.

        Args:
            data (str): Data to run.

        Returns:
            str: Output data.
        """
        data = self._preprocess(data)
        inference_data = []
        for d in data:
            inference_data.append(
                self._client.chat.completions.create(
                    model=self._model_config.model_name,
                    messages=[
                        {"role": "user", "content": d},
                    ],
                    n=self._model_config.num_call,
                    temperature=self._model_config.temperature,
                    response_format=self._model_config.response_format,
                )
            )
        data = self._postprocess(inference_data)
        return data


class HuggingfaceModelServer(AbsModelServer):
    """Huggingface Model Server Class."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        # import in class level to avoid installing transformers package
        from transformers import pipeline  # pylint: disable=import-outside-toplevel
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        super().__init__(model_config)
        self._model_config = HuggingfaceModelConfig(**self._model_config)

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.model_name,
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self._model_config.model_name,
            device_map="auto",
            offload_folder="./offload",
            load_in_4bit=True,
        )

        # explicitly set batch_size for pipeline
        # for batch inference.
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=768,
            num_return_sequences=1,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=self._model_config.batch_size,
        )

    def _preprocess(self, data: List[str]) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (List[str]): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return [d["generated_text"] for output_list in data for d in output_list]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Args:
            data (List[str]): Data to run.

        Returns:
            List[str]: Output data.
        """
        data = self._preprocess(data)
        data = self._pipeline(data)
        data = self._postprocess(data)
        return data


class LMQGModelServer(AbsModelServer):
    """Huggingface Model Server Class."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        # import in class level to avoid installing transformers package
        from lmqg import TransformersQG  # pylint: disable=import-outside-toplevel

        super().__init__(model_config)
        self._model_config = LMQGModelConfig(**self._model_config)

        self._model = TransformersQG(
            model=self._model_config.model_name, max_length=1024
        )

    def _preprocess(self, data: List[str]) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (List[str]): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return data

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Args:
            data (List[str]): Data to run.

        Returns:
            List[str]: Output data.
        """
        data = self._preprocess(data)
        data = self._model.generate_qa(data)
        data = self._postprocess(data)
        return data


class NougatModelServer(AbsModelServer):
    """Nougat Model Server Class."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        # import in class level to avoid installing nougat package
        try:
            from nougat import NougatModel  # pylint: disable=import-outside-toplevel
            from nougat.utils.checkpoint import (  # pylint: disable=import-outside-toplevel
                get_checkpoint,
            )
            from nougat.utils.device import (  # pylint: disable=import-outside-toplevel
                move_to_device,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install nougat to use NougatModelServer. You can use `pip install nougat-ocr` to install it."
            ) from exc

        super().__init__(model_config)
        self._model_config = NougatModelConfig(**self._model_config)
        checkpoint = get_checkpoint(None, model_tag=self._model_config.model_name)
        self.model = NougatModel.from_pretrained(checkpoint)
        self.model = move_to_device(
            self.model, bf16=False, cuda=self._model_config.batch_size > 0
        )
        self.model.eval()

    def _preprocess(self, data: str) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (List[str]): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return [d["generated_text"] for output_list in data for d in output_list]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Args:
            data (List[str]): Data to run.

        Returns:
            List[str]: Output data.
        """
        from nougat.postprocessing import (  # pylint: disable=import-outside-toplevel
            markdown_compatible,
        )
        from nougat.utils.dataset import (  # pylint: disable=import-outside-toplevel
            LazyDataset,
        )
        from torch.utils.data import (  # pylint: disable=import-outside-toplevel
            ConcatDataset,
            DataLoader,
        )

        outs = []
        for pdf in data:
            dataset = LazyDataset(
                pdf,
                partial(self.model.encoder.prepare_input, random_padding=False),
                None,
            )
            dataloader = DataLoader(
                ConcatDataset([dataset]),
                batch_size=1,
                shuffle=False,
                collate_fn=LazyDataset.ignore_none_collate,
            )
            predictions = []
            page_num = 0
            for i, (sample, is_last_page) in enumerate(dataloader):
                model_output = self.model.inference(
                    image_tensors=sample, early_stopping=False
                )
                # check if model output is faulty
                for j, output in enumerate(model_output["predictions"]):
                    page_num += 1
                    if output.strip() == "[MISSING_PAGE_POST]":
                        # uncaught repetitions -- most likely empty page
                        predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                    else:
                        output = markdown_compatible(output)
                        predictions.append(output)
                    if is_last_page[j]:
                        out = "".join(predictions).strip()
                        out = re.sub(r"\n{3,}", "\n\n", out).strip()
            outs.append(out)
        return outs
