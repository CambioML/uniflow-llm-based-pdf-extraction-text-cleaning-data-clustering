"""
All Model Servers including ModelServerFactory, AbsModelServer, OpenAIModelServer and HuggingfaceModelServer.
"""

import abc
import json
import logging
import re
import warnings
from functools import partial
from typing import Any, Dict, List, Optional

from uniflow.op.model.model_config import (
    AzureOpenAIModelConfig,
    BedrockModelConfig,
    HuggingfaceModelConfig,
    LayoutModelConfig,
    LMQGModelConfig,
    NougatModelConfig,
    OpenAIModelConfig,
    SageMakerModelConfig,
)
from uniflow.op.prompt import PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize AbsModelServer class.

        Args:
            prompt_template (PromptTemplate): Prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        self._model_config = model_config
        self._example_keys = None
        if prompt_template.few_shot_prompt:
            self._example_keys = list(
                prompt_template.few_shot_prompt[0].model_dump().keys()
            )

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

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        # import in class level to avoid installing openai package
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        super().__init__(prompt_template, model_config)
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

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        # import in class level to avoid installing openai package
        from openai import AzureOpenAI  # pylint: disable=import-outside-toplevel

        super().__init__(prompt_template, model_config)
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

    PATTERN = r"\[\/?INST\]|<s>|<<SYS>>|\[ASST\]|\[\/ASST\]"

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        # import in class level to avoid installing transformers package
        super().__init__(prompt_template, model_config)
        self._model_config = HuggingfaceModelConfig(**self._model_config)
        if self._model_config.neuron is False:
            try:
                from transformers import (  # pylint: disable=import-outside-toplevel
                    pipeline,
                )
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Please install transformers to use HuggingfaceModelServer. You can use `pip install transformers` to install it."
                ) from exc
            model, tokenizer = self._get_model()
            # explicitly set batch_size for pipeline
            # for batch inference.
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                do_sample=self._model_config.do_sample,
                temperature=self._model_config.temperature,
                num_beams=self._model_config.num_beams,
                max_new_tokens=self._model_config.max_new_tokens,
                num_return_sequences=self._model_config.num_return_sequences,
                repetition_penalty=self._model_config.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                batch_size=self._model_config.batch_size,
            )
        else:
            if self._model_config.load_in_4bit or self._model_config.load_in_8bit:
                self._model_config.load_in_4bit = False
                self._model_config.load_in_8bit = False
                print(
                    "Neuron model does not support quantized models. load_in_4bit and load_in_8bit are automatically set to False."
                )
            from uniflow.op.model.neuron_utils import (  # pylint: disable=import-outside-toplevel
                Neuron,
            )

            model, tokenizer = Neuron.get_neuron_model(
                self._model_config.model_name, self._model_config.batch_size
            )
            self._pipeline = partial(
                Neuron.neuron_infer, model=model, tokenizer=tokenizer
            )
        self._tokenizer = tokenizer

    def _get_model(self):
        """Get model."""
        from transformers import (  # pylint: disable=import-outside-toplevel
            AutoModelForCausalLM,
            AutoTokenizer,
        )

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
        return model, tokenizer

    def _preprocess(self, data: List[str]) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        # add role and content key to data for apply_chat_template
        # as argument
        data = [[{"role": "user", "content": d}] for d in data]
        # if response_start_key is provided (few shot mode), add it with colon after
        # the end of instruction token for better instruction following performance.
        # Below is an example, if you have a QA prompt template like this for 1 shot mode:

        # <s>[INST] "instruction: This is an instruction.\n <-- instruction
        # context: ... <-- few shot context
        # question: ... <-- few shot question
        # answer: ... <-- few shot answer
        # context: ... [/INST] <-- input context with [/INST]
        # question:   <-- response_start_key is added here !!!
        if self._model_config.response_start_key:
            data = [
                self._tokenizer.apply_chat_template(d, tokenize=False)
                + f"\n{self._model_config.response_start_key}: "  # noqa: W503
                for d in data
            ]
        # if response_start_key is not provided, simply add the instruction token
        # using apply_chat_template
        else:
            data = [
                self._tokenizer.apply_chat_template(d, tokenize=False) for d in data
            ]
        return data

    def _postprocess(self, data: List[str]) -> List[str]:
        """Postprocess data.

        Args:
            data (List[str]): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        response_list = []
        # clean up instruction token.
        for output_list in data:
            for d in output_list:
                response = re.sub(self.PATTERN, "", d["generated_text"]).strip()
                response_list.append(response)

        # if response_format is json_object, parse the response into json_object.
        if (
            self._model_config.response_format
            and self._model_config.response_format["type"]  # noqa: W503
            == "json_object"  # noqa: W503
        ):
            # if example_keys (through few shot prompt) are provided,
            # parse the response into json_object.
            if self._example_keys:
                keywords = [f"{example_key}:" for example_key in self._example_keys]
                pattern = "|".join(map(re.escape, keywords))
                json_response_list = []
                for response in response_list:
                    segments = [
                        segment.strip() for segment in re.split(pattern, response)
                    ]
                    offset = len(segments) - len(self._example_keys)
                    result_dict = {
                        key: value
                        for key, value in zip(self._example_keys, segments[offset:])
                    }

                    json_response_list.append(result_dict)
                    response_list = json_response_list
            else:
                # if example_keys are not provided, simply return the raw response
                # even if response_format is json_object. This is because without
                # few shot prompt, model is not stable to generate parsed response
                # into json_object.
                logging.info(
                    "No example keys found in the prompt template. Returning the raw response without json_object format."
                )

        return response_list

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

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        # import in class level to avoid installing transformers package
        from lmqg import TransformersQG  # pylint: disable=import-outside-toplevel

        super().__init__(prompt_template, model_config)
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

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
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

        super().__init__(prompt_template, model_config)
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
            for sample, is_last_page in dataloader:
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


class AWSBaseModelServer(AbsModelServer):
    """AWS Base Model Server Class."""

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        try:
            # import in class level to avoid installing boto3
            import boto3

            super().__init__(prompt_template, model_config)

            # If user specifies profile in model config, use that profile
            if "aws_profile" in model_config:
                aws_profile = model_config.get("aws_profile", "default")
                self._session = boto3.Session(profile_name=aws_profile)
            # Otherwise if the user specifies credentials directly in the model config, use those credentials
            elif model_config.get("aws_access_key_id") and model_config.get(
                "aws_secret_access_key"
            ):
                self._session = boto3.Session(
                    aws_access_key_id=model_config.get("aws_access_key_id"),
                    aws_secret_access_key=model_config.get("aws_secret_access_key"),
                    aws_session_token=model_config.get("aws_session_token"),
                )
                warnings.warn(
                    "Using AWS credentials directly in the model config is not recommended. "
                    "Please use a profile instead."
                )
            else:
                self._session = boto3.Session(profile_name="default")
                warnings.warn(
                    "Using default profile to create the session. "
                    "Please pass the profile name in the model config."
                )

            self.aws_region = model_config.get("aws_region", None)

        except ImportError as exc:
            raise ModuleNotFoundError(
                "Failed to import the 'boto3' Python package. "
                "Please install it by running `pip install boto3`."
            ) from exc
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

    def enforce_stop_tokens(self, text: str, stop: List[str]) -> str:
        """Cut off the text as soon as any stop words occur."""
        return re.split("|".join(stop), text, maxsplit=1)[0]

    @abc.abstractmethod
    def prepare_input(
        self, provider: str, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare the input for the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_output(self, provider: str, response: Any) -> str:
        """
        Prepares the output based on the provider and response.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, data: List[str]) -> List[str]:
        """
        Run model.
        """
        raise NotImplementedError


class BedrockModelServer(AWSBaseModelServer):
    """Bedrock Model Server Class.

    The AWS client authenticates by automatically loading credentials as per the methods outlined here:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If you wish to use a specific credential profile, please provide the profile name from your ~/.aws/credentials file.

    Make sure that the credentials or roles in use have the necessary policies for Bedrock service access.

    Additionally, it is important to verify that your boto3 version supports the Bedrock runtime.
    """

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        super().__init__(prompt_template, model_config)
        self._model_config = BedrockModelConfig(**self._model_config)
        self._client = self._session.client(
            "bedrock-runtime", region_name=self.aws_region
        )

    def _get_provider(self) -> str:
        return self._model_config.model_name.split(".")[0]

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
            raise ValueError(f"Error raised by bedrock service: {e}") from e

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


class SageMakerModelServer(AWSBaseModelServer):
    """
    SageMaker Model Server Class.

    The AWS client authenticates by automatically loading credentials as per the methods outlined here:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If you wish to use a specific credential profile, please provide the profile name from your ~/.aws/credentials file.

    Make sure that the credentials or roles in use have the necessary policies for SageMaker service access.
    """

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        super().__init__(prompt_template, model_config)
        self._model_config = SageMakerModelConfig(**self._model_config)
        self._client = self._session.client(
            "sagemaker-runtime", region_name=self.aws_region
        )

    def prepare_input(
        self, model_type: str, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare the input for the model based on the model_type.

        Args:
            model_type (str): The type of the model.
            prompt (str): The input prompt.
            model_kwargs (Dict[str, Any]): Additional model arguments.

        Returns:
            Dict[str, Any]: The prepared input for the model.
        """

        def prepare_falcon_input(
            prompt: str, model_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            input_body = {
                "inputs": f"{prompt}",
                "parameters": model_kwargs,
            }
            return input_body

        def prepare_mistral_input(
            prompt: str, model_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            input_body = {"inputs": prompt, "parameters": model_kwargs}
            return input_body

        model_input_preparation = {
            "falcon": prepare_falcon_input,
            "mistral": prepare_mistral_input,
        }

        prepare_input_for_model = model_input_preparation.get(
            model_type, prepare_mistral_input
        )
        return prepare_input_for_model(prompt, model_kwargs)

    def prepare_output(self, model_type: str, response: Any) -> str:
        """
        Prepares the output based on the model_type and response.

        Args:
            model_type (str): The model_type of the response.
            response (Any): The response object.

        Returns:
            str: The prepared output.

        Raises:
            None
        """

        def prepare_falcon_output(response: Any) -> str:
            response_body = json.loads(response.get("Body").read())
            return response_body[0].get("generated_text")

        def prepare_mistral_output(response: Any) -> str:
            response_body = json.loads(response.get("Body").read())
            return response_body.get("outputs")

        model_output_preparation = {
            "falcon": prepare_falcon_output,
            "mistral": prepare_mistral_output,
        }

        prepare_output_for_model = model_output_preparation.get(
            model_type, prepare_mistral_output
        )
        return prepare_output_for_model(response)

    def invoke_sagemaker_model(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Invokes the sagemaker model with the given prompt and optional stop tokens.

        Args:
            prompt (str): The input prompt for the model.
            stop (Optional[List[str]]): List of stop tokens to indicate the end of the generated text.
            **kwargs: Additional keyword arguments to be passed to the model.

        Returns:
            str: The generated text from the sagemaker model.

        Raises:
            ValueError: If there is an error raised by the Amazon Sagemaker service.
        """
        model_type = self._model_config.model_type
        _model_kwargs = self._model_config.model_kwargs or {}
        params = {**_model_kwargs, **kwargs}

        # Combine the prompt and model parameters into a single input body
        input_body = self.prepare_input(model_type, prompt, params)
        body = json.dumps(input_body)
        accept = "application/json"
        content_type = "application/json"

        # Invoke the model
        try:
            response = self._client.invoke_endpoint(
                EndpointName=self._model_config.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accept,
            )
        except Exception as e:
            raise ValueError(f"Error raised by sagemaker service: {e}") from e

        # Perform post-processing on the response
        text = self.prepare_output(model_type, response)

        if stop is not None:
            text = self.enforce_stop_tokens(text, stop)

        return text

    def __call__(self, data: List[str]) -> List[str]:
        """Run model.

        Args:
            data List[str]: Data to run.

        Returns:
            str: Output data.
        """
        data = self._preprocess(data)
        inference_data = []
        for d in data:
            inference_data.append(self.invoke_sagemaker_model(prompt=d))
        data = self._postprocess(inference_data)
        return data


class LayoutModelServer(AbsModelServer):
    """Layout Model Server Class."""

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:
        super().__init__(prompt_template, model_config)
        self._model_config = LayoutModelConfig(**self._model_config)
        try:
            import easyocr  # pylint: disable=import-outside-toplevel

            self.reader = easyocr.Reader(self._model_config.ocr_lang)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install easyocr to use LayoutModelServer. You can use `pip install easyocr` to install it."
            ) from exc
        from .layout_utils import (  # pylint: disable=import-outside-toplevel
            LayoutPredictor,
        )

        self.layout_predictor = LayoutPredictor(
            self._model_config.model_name, self._model_config.model_file
        )

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
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        from uniflow.op.model.layout_utils import (  # pylint: disable=import-outside-toplevel
            XYCut,
        )

        outs = []
        for img in data:
            img = cv2.imread(img)
            ori_im = img.copy()
            h, w, _ = img.shape
            layout_res = self.layout_predictor(img)
            res_list = []
            for region in layout_res:
                res = ""
                if region["bbox"] is not None:
                    x1, y1, x2, y2 = region["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)

                wht_im[y1:y2, x1:x2, :] = roi_img
                result = self.reader.readtext(wht_im)
                if len(result) == 0:
                    continue
                filter_boxes, filter_rec_res, scores = zip(*result)
                res = []
                for box, rec_res, score in zip(filter_boxes, filter_rec_res, scores):
                    rec_str = rec_res
                    rec_conf = score
                    res.append(
                        {
                            "text": rec_str,
                            "confidence": float(rec_conf),
                            "text_region": box,
                        }
                    )
                res_list.append(
                    {
                        "type": region["type"].lower(),
                        "bbox": [x1, y1, x2, y2],
                        "img": roi_img,
                        "res": res,
                    }
                )
            res = []
            boxes = [res["bbox"] for res in res_list]
            XYCut.recursive_xy_cut(
                np.asarray(boxes).astype(int), np.arange(len(boxes)), res
            )
            sorted_res_list = [res_list[idx] for idx in res]
            final_md = ""
            for _, region in enumerate(sorted_res_list):
                if len(region["res"]) == 0:
                    continue
                if region["type"] in ("title", "page-header", "section-header"):
                    final_md += (
                        "## "
                        + " ".join([text["text"] for text in region["res"]])
                        + "\n\n"
                    )
                elif region["type"] in (
                    "picture",
                    "footnote",
                    "formula",
                    "list-item",
                    "text",
                    "caption",
                    "page-footer",
                    "table",
                ):
                    final_md += (
                        " ".join([text["text"] for text in region["res"]]) + "\n\n"
                    )
                else:
                    print(region["type"])
            out = re.sub(r"\n{3,}", "\n\n", final_md.strip()).strip()
            outs.append(out)
        return outs
