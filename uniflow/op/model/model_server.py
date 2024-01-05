"""
All Model Servers including ModelServerFactory, AbsModelServer, OpenAIModelServer and HuggingfaceModelServer.
"""

import json
import re
import warnings
from functools import partial
from typing import Any, Dict, List, Optional

from uniflow.op.model.model_config import (
    AzureOpenAIModelConfig,
    BedrockModelConfig,
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


class AzureOpenAIModelServer(AbsModelServer):
    """Azure OpenAI Model Server Class."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        # import in class level to avoid installing openai package
        from openai import AzureOpenAI  # pylint: disable=import-outside-toplevel

        super().__init__(model_config)
        self._model_config = AzureOpenAIModelConfig(**self._model_config)
        self._client = AzureOpenAI(
            api_key=self._model_config.api_key,
            api_version=self._model_config.api_version,
            azure_endpoint=self._model_config.azure_endpoint,
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
            data (str): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        return [c.message.content for d in data for c in d.choices]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Azure OpenAI completions API does not support batch inference.

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
            load_in_4bit=self._model_config.load_in_4bit,
            load_in_8bit=self._model_config.load_in_8bit,
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


class BedrockModelServer(AbsModelServer):
    """Bedrock Model Server Class.

    The AWS client authenticates by automatically loading credentials as per the methods outlined here:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If you wish to use a specific credential profile, please provide the profile name from your ~/.aws/credentials file.

    Make sure that the credentials or roles in use have the necessary policies for Bedrock service access.

    Additionally, it is important to verify that your boto3 version supports the Bedrock runtime.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        try:
            # import in class level to avoid installing boto3
            import boto3

            super().__init__(model_config)
            self._model_config = BedrockModelConfig(**self._model_config)

            # If user specifies profile in model config, use that profile
            if "aws_profile" in model_config:
                aws_profile = self._model_config.aws_profile
                session = boto3.Session(profile_name=aws_profile)
            # Otherwise if the user specifies credentials directly in the model config, use those credentials
            elif (
                self._model_config.aws_access_key_id
                and self._model_config.aws_secret_access_key
            ):
                session = boto3.Session(
                    aws_access_key_id=self._model_config.aws_access_key_id,
                    aws_secret_access_key=self._model_config.aws_secret_access_key,
                    aws_session_token=self._model_config.aws_session_token,
                )
                warnings.warn(
                    "Using AWS credentials directly in the model config is not recommended. "
                    "Please use a profile instead."
                )
            else:
                session = boto3.Session(profile_name="default")
                warnings.warn(
                    "Using default profile to create the session. "
                    "Please pass the profile name in the model config."
                )

            aws_region = (
                self._model_config.aws_region if self._model_config.aws_region else None
            )

            self._client = session.client("bedrock-runtime", region_name=aws_region)

        except ImportError:
            raise ModuleNotFoundError(
                "Failed to import the 'boto3' Python package. "
                "Please install it by running `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Failed to load credentials for authenticating with the AWS client. "
                "Please ensure that the specified profile name contains valid credentials."
            ) from e

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
        return data

    def _get_provider(self) -> str:
        return self._model_config.model_name.split(".")[0]

    def enforce_stop_tokens(self, text: str, stop: List[str]) -> str:
        """Cut off the text as soon as any stop words occur."""
        return re.split("|".join(stop), text, maxsplit=1)[0]

    def prepare_input(
        self, provider: str, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare the input for the model based on the provider.

        Args:
            provider (str): The provider of the model.
            prompt (str): The input prompt.
            model_kwargs (Dict[str, Any]): Additional model arguments.

        Returns:
            Dict[str, Any]: The prepared input for the model.
        """

        def prepare_anthropic_input(
            prompt: str, model_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            input_body = {
                **model_kwargs,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant: ",
            }
            if "max_tokens_to_sample" not in input_body:
                input_body["max_tokens_to_sample"] = 256
            return input_body

        def prepare_ai21_cohere_meta_input(
            prompt: str, model_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {**model_kwargs, "prompt": prompt}

        def prepare_amazon_input(
            prompt: str, model_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {"inputText": prompt, "textGenerationConfig": {**model_kwargs}}

        def prepare_default_input(
            prompt: str, model_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {"inputText": prompt}

        provider_input_preparation = {
            "anthropic": prepare_anthropic_input,
            "ai21": prepare_ai21_cohere_meta_input,
            "cohere": prepare_ai21_cohere_meta_input,
            "meta": prepare_ai21_cohere_meta_input,
            "amazon": prepare_amazon_input,
        }

        prepare_input_for_provider = provider_input_preparation.get(
            provider, prepare_default_input
        )
        return prepare_input_for_provider(prompt, model_kwargs)

    def prepare_output(self, provider: str, response: Any) -> str:
        """
        Prepares the output based on the provider and response.

        Args:
            provider (str): The provider of the response.
            response (Any): The response object.

        Returns:
            str: The prepared output.

        Raises:
            None
        """

        def prepare_anthropic_output(response: Any) -> str:
            response_body = json.loads(response.get("body").read().decode())
            return response_body.get("completion")

        def prepare_ai21_output(response: Any) -> str:
            response_body = json.loads(response.get("body").read())
            return response_body.get("completions")[0].get("data").get("text")

        def prepare_cohere_output(response: Any) -> str:
            response_body = json.loads(response.get("body").read())
            return response_body.get("generations")[0].get("text")

        def prepare_meta_output(response: Any) -> str:
            response_body = json.loads(response.get("body").read())
            return response_body.get("generation")

        def prepare_default_output(response: Any) -> str:
            response_body = json.loads(response.get("body").read())
            return response_body.get("results")[0].get("outputText")

        provider_output_preparation = {
            "anthropic": prepare_anthropic_output,
            "ai21": prepare_ai21_output,
            "cohere": prepare_cohere_output,
            "meta": prepare_meta_output,
        }

        prepare_output_for_provider = provider_output_preparation.get(
            provider, prepare_default_output
        )
        return prepare_output_for_provider(response)

    def invoke_bedrock_model(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Invokes the bedrock model with the given prompt and optional stop tokens.

        Args:
            prompt (str): The input prompt for the model.
            stop (Optional[List[str]]): List of stop tokens to indicate the end of the generated text.
            **kwargs: Additional keyword arguments to be passed to the model. Please refer to
            https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html for more details.

        Returns:
            str: The generated text from the bedrock model.

        Raises:
            ValueError: If there is an error raised by the bedrock service.
        """
        provider = self._get_provider()
        _model_kwargs = self._model_config.model_kwargs or {}
        params = {**_model_kwargs, **kwargs}

        # Combine the prompt and model parameters into a single input body
        input_body = self.prepare_input(provider, prompt, params)
        body = json.dumps(input_body)
        accept = "application/json"
        contentType = "application/json"

        # Invoke the model
        try:
            response = self._client.invoke_model(
                body=body,
                modelId=self._model_config.model_name,
                accept=accept,
                contentType=contentType,
            )
        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        # Perform post-processing on the response
        text = self.prepare_output(provider, response)

        if stop is not None:
            text = self.enforce_stop_tokens(text, stop)

        return text

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Current bedrock batch inference is implemented by creating asynchronous jobs.
        At present, we are not temporarily using Batch Inference.
        Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-create.html

        Args:
            data List[str]: Data to run.

        Returns:
            str: Output data.
        """
        data = self._preprocess(data)
        inference_data = []
        for d in data:
            inference_data.append(self.invoke_bedrock_model(prompt=d))
        data = self._postprocess(inference_data)
        return data
