# Examples
## Table of Contents
1. [Base Config](#base-config)
    - [Examples](#examples-1)
1. [OpenAIConfig](#openaiconfig)
    - [Examples](#examples-2)
1. [HuggingfaceConfig](#huggingfaceconfig)
    - [Examples](#examples-3)
1. [LMQGConfig](#lmqgconfig)
    - [Examples](#examples-4)

## Base Config
The base `Config` is the base configuration that all other configurations inherit from. Here are the default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | `ModelFlow` | The name of the flow to run. |
| `guided_prompt_template` | `GuidedPrompt` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `ModelConfig` | The model configuration to use. |

Here are the default parameters for the `ModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |

### Examples
Please refer to these example notebooks for more details:

| Notebook | Description |
| --- | --- |
| [model.ipynb](./model.ipynb) | Basic example of using the `Config` with the `OpenAIModelConfig` to generate QA on dummy samples |

## OpenAIConfig
The `OpenAIConfig` configuration runs the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | `OpenAIModelFlow` | The name of the flow to run. |
| `guided_prompt_template` | `GuidedPrompt` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `OpenAIModelConfig` | The model configuration to use. |

Here are the default parameters for the `OpenAIModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |
| `num_call` | `int` | 1 | The number of calls to make to the OpenAI model |
| `temperature` | `float` | 1.5 | The temperature to use for the OpenAI model. |
| `response_format` | `Dict[str, str]` | {"type": "text"} | The response format to use for the OpenAI model. |

### Examples
Please refer to these example notebooks for more details:

| Notebook | Description |
| --- | --- |
| [openai_model.ipynb](./openai_model.ipynb) | Generates QA using the `uniflow` defaults on dummy sample text. |
| [openai_json_model.ipynb](./openai_json_model.ipynb) | Generates QA with a custom instruction on dummy sample text. Uses the `json_object` response format. |
| [openai_pdf_source_10k_QA.ipynb](./openai_pdf_source_10k_QA.ipynb) | Generates QA with a custom few-shot example on a Nike 10K pdf. Uses the `json_object` response format. Saves results to a csv. |
| [openai_pdf_source_10k_summary.ipynb](./openai_pdf_source_10k_summary.ipynb) | Generates summaries with a custom instruction and few-shot example on a Nike 10K pdf. Uses the `json_object` response format. Saves results to a csv. |

## HuggingfaceConfig
The `HuggingfaceConfig` configuration has the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | `HuggingfaceModelFlow`| The name of the flow to run. |
| `guided_prompt_template` | `GuidedPrompt` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `HuggingfaceModelConfig` | The model configuration to use. |

Here are the default parameters for the `HuggingfaceModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `mistralai/Mistral-7B-Instruct-v0.1` | The name of the model to use. |
| `batch_size` | `int` | 1 | The batch size to use for the Huggingface model. |

### Examples
Please refer to these example notebooks for more details:

| Notebook | Description |
| --- | --- |
| [huggingface_model.ipynb](./huggingface_model.ipynb)  | Generates QAs using `HuggingfaceConfig` with a custom instruction and examples for dummy sample text. Demonstrates use of `batch_size` configuration. |

## LMQGConfig
Unlike the other models, LMQG does not use a `GuidedPrompt`, but instead only passes in contexts. This is because the `LMQG` model already generates questionsa and answers by default.

The `LMQGConfig` configuration runs with the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | `LMQGModelFlow` | The name of the flow to run. |
| `guided_prompt_template` | `GuidedPrompt` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `LMQGModelConfig` | The model configuration to use. |

Here are the default parameters for the `LMQGModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `lmqg/t5-base-squad-qg-ae` | The name of the model to use. |
| `batch_size` | `int` | 1 | The batch size to use for the LMQG model. |

See the  for a working example.

### Examples
Please refer to these example notebooks for more details:

| Notebook | Description |
| --- | --- |
| [lmqg_model.ipynb](./lmqg_model.ipynb) | Generates QA with `LMQGConfig` using the `uniflow` defaults on dummy sample text. |
