"""
All Model Servers including ModelServerFactory, AbsModelServer, OpenAIModelServer and HuggingfaceModelServer.
"""

import re
from functools import partial
from typing import Any, Dict, List

from uniflow.op.model.model_config import (
    HuggingfaceModelConfig,
    LMQGModelConfig,
    NougatModelConfig,
    OpenAIModelConfig,
)

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
            from nougat.utils.checkpoint import (
                get_checkpoint,  # pylint: disable=import-outside-toplevel
            )
            from nougat.utils.device import (
                move_to_device,  # pylint: disable=import-outside-toplevel
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
        from nougat.postprocessing import (
            markdown_compatible,  # pylint: disable=import-outside-toplevel
        )
        from nougat.utils.dataset import (
            LazyDataset,  # pylint: disable=import-outside-toplevel
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
