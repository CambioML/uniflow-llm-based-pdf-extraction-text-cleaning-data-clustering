# Examples
## Base Config
The base `Config` is the base configuration that all other configurations inherit from. Here are the default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [ModelFlow] | The name of the flow to run. |
| `prompt_template` | `PromptTemplate` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `ModelConfig` | The model configuration to use. |

Here are the default parameters for the `ModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |

The [model.ipynb notebook](./model.ipynb) shows a basic example of how to use the base `Config`, where it also passes the `OpenAIModelConfig` as a `model_config` argument.

## OpenAIConfig
The `OpenAIConfig` configuration runs the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | `OpenAIModelFlow` | The name of the flow to run. |
| `prompt_template` | `PromptTemplate` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `OpenAIModelConfig` | The model configuration to use. |

Here are the default parameters for the `OpenAIModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |
| `num_call` | `int` | 1 | The number of calls to make to the OpenAI model |
| `temperature` | `float` | 1.5 | The temperature to use for the OpenAI model. |
| `response_format` | `Dict[str, str]` | {"type": "text"} | The response format to use for the OpenAI model. |

See the [openai_json_model.ipynb notebook](./openai_json_model.ipynb) for a working example.

## HuggingfaceConfig
The `HuggingfaceConfig` configuration has the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [HuggingfaceModelFlow](../../README.md#model) | The name of the flow to run. |
| `prompt_template` | `PromptTemplate` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `HuggingfaceModelConfig` | The model configuration to use. |

Here are the default parameters for the `HuggingfaceModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `mistralai/Mistral-7B-Instruct-v0.1` | The name of the model to use. |
| `batch_size` | `int` | 1 | The batch size to use for the Huggingface model. |

See the [huggingface_model.ipynb notebook](./huggingface_model.ipynb) for a working example.

## LMQGModelConfig
The `LMQGModelConfig` configuration runs with the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | `LMQGModelFlow` | The name of the flow to run. |
| `prompt_template` | `PromptTemplate` | [Default](../../README.md#2-prompting) | The template to use for the guided prompt. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `LMQGModelConfig` | The model configuration to use. |

Here are the default parameters for the `LMQGModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `lmqg/t5-base-squad-qg-ae` | The name of the model to use. |
| `batch_size` | `int` | 1 | The batch size to use for the LMQG model. |

See the [lmqg_model.ipynb notebook](./lmqg_model.ipynb) for a working example.
