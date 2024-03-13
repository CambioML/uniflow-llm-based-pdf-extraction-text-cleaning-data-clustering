"""Model config module."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Model Config Class."""

    model_name: str = "gpt-3.5-turbo-1106"
    model_server: str = "OpenAIModelServer"


@dataclass
class GoogleModelConfig(ModelConfig):
    """Google Model Config Class."""

    model_name: str = "models/gemini-1.0-pro-001"
    model_server: str = "GoogleModelServer"
    max_output_tokens: int = 2048
    temperature: float = 0
    top_k: int = 1
    top_p: float = 1.0
    candidate_count: int = 1
    num_thread: int = 1


@dataclass
class GoogleMultiModalModelConfig(GoogleModelConfig):
    """Google MultiModal Model Config Class."""

    model_name: str = "gemini-pro-vision"
    model_server: str = "GoogleMultiModalModelServer"


@dataclass
class OpenAIModelConfig(ModelConfig):
    """OpenAI Model Config Class."""

    model_server: str = "OpenAIModelServer"
    num_call: int = 1
    temperature: float = 0.9
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})
    num_thread: int = 1


@dataclass
class AzureOpenAIModelConfig:
    """Azure OpenAI Model Config Class."""

    api_key: str = os.getenv("AZURE_API_KEY")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT")
    api_version: str = os.getenv("AZURE_API_VERSION")
    model_name: str = os.getenv("AZURE_DEPLOYMENT_NAME")
    model_server: str = "AzureOpenAIModelServer"
    num_call: int = 1
    temperature: float = 0.7
    response_format: Dict[str, str] = field(default_factory=lambda: {"type": "text"})
    num_thread: int = 1


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

    model_name: str = "facebook/nougat-small"
    batch_size: int = 2
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
