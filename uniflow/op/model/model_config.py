"""Model config module."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Model Config Class."""

    model_name: str = "gpt-3.5-turbo-1106"
    model_server: str = "OpenAIModelServer"


@dataclass
class OpenAIModelConfig(ModelConfig):
    """OpenAI Model Config Class."""

    model_server: str = "OpenAIModelServer"
    num_call: int = 1
    temperature: float = 0.9
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})


@dataclass
class AzureOpenAIModelConfig:
    """Azure OpenAI Model Config Class."""

    api_key: str
    api_version: str
    azure_endpoint: str
    model_name: str = "gpt-3.5-turbo-1106"
    model_server: str = "AzureOpenAIModelServer"
    num_call: int = 1
    temperature: float = 0.9
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})


@dataclass
class HuggingfaceModelConfig(ModelConfig):
    """Huggingface Model Config Class."""

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    batch_size: int = 1
    model_server: str = "HuggingfaceModelServer"
    neuron: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = True
    max_new_tokens: int = 768
    do_sample: bool = False
    temperature: float = 0.0
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.2
    response_start_key: str = None
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})


@dataclass
class LMQGModelConfig(ModelConfig):
    """LMQG Model Config Class."""

    model_name: str = "lmqg/t5-base-squad-qg-ae"
    batch_size: int = 1
    model_server: str = "LMQGModelServer"


@dataclass
class NougatModelConfig(ModelConfig):
    """Nougat Model Config Class."""

    model_name: str = "0.1.0-small"
    batch_size: int = 1
    model_server: str = "NougatModelServer"


@dataclass
class LayoutModelConfig(ModelConfig):
    """Layout Model Config Class."""

    model_name: str = "unstructuredio/yolo_x_layout"
    model_file: str = "yolox_l0.05.onnx"
    model_server: str = "LayoutModelServer"
    ocr_lang: list = field(default_factory=lambda: ["en"])


@dataclass
class BedrockModelConfig:
    """Bedrock Model Config Class."""

    aws_region: str = ""
    aws_profile: str = "default"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    model_name: str = "anthropic.claude-v2"
    batch_size: int = 1
    model_server: str = "BedrockModelServer"
    # TODO: Need to consider the best approach for handling model arguments
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class SageMakerModelConfig:
    """SageMaker Model Config Class."""

    endpoint_name: str
    model_type: str
    aws_region: str = "us-west-2"
    aws_profile: str = "default"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    model_server: str = "SageMakerModelServer"
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
