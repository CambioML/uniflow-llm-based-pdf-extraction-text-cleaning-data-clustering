# Config Examples
## Base Config
The base `Config` is the base configuration that all other configurations inherit from. Here are the default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [ModelFlow](../../README.md#jsonmodel) | The name of the flow to run. |
| `few_shot_template` | `List[Dict[str, str]]` | {} | The template to use for the few-shot model. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `ModelConfig` | The model configuration to use. |

Here are the default parameters for the `ModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |

The [model.ipynb notebook](./model.ipynb) shows a basic example of how to use the `BaseConfig`, where it also passes the `OpenAIModelConfig` as a `model_config` argument.

## OpenAIFewShotConfig
The `OpenAIFewShotConfig` has the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [FewShotModelFlow](../../README.md#fewshotmodel) | The name of the flow to run. |
| `few_shot_template` | `List[Dict[str, str]]` | See below | The template to use for the few-shot model. |
| `num_threads` | `int` | 1 | The number of threads to use. |
`model_config` | `ModelConfig` | `OpenAIModelConfig` | The model configuration to use. |

Here is the default `few_shot_template`:
```
{
            "instruction": """Generate one question and its corresponding answer based on the context.
    Following the format of the examples below. The response must include original context, question, and answer.""",
            "examples": [
                {
                    "context": """The quick brown fox jumps over the lazy dog.""",
                    "question": """What is the color of the fox?""",
                    "answer": """brown.""",
                },
                {
                    "context": """The quick brown fox jumps over the lazy black dog.""",
                    "question": """What is the color of the dog?""",
                    "answer": """black.""",
                },
            ],
        }
```

The `data` you pass in is appended to the `examples` list of the template.

Here are the default parameters for the `OpenAIModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |
| `num_call` | `int` | 1 | The number of calls to make to the OpenAI model |
| `temperature` | `float` | 1.5 | The temperature to use for the OpenAI model. |
| `response_format` | `Dict[str, str]` | {"type": "text"} | The response format to use for the OpenAI model. |

 The [few_shot_model.ipynb notebook](./few_shot_model.ipynb) shows a basic example of how to use the `OpenAIFewShotConfig`.

 The [self_instruct_few_shot_html_source.ipynb notebook](./self_instruct_few_shot_html_source.ipynb) shows an example of how to use the `OpenAIFewShotConfig` for a self-instructed model.

## OpenAIJsonConfig
The `OpenAIJsonConfig` configuration runs the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [OpenAIJsonModelFlow](../../README.md#jsonmodel) | The name of the flow to run. |
| `few_shot_template` | `List[Dict[str, str]]` | {} | The template to use for the few-shot model. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `OpenAIJsonModelConfig` | The model configuration to use. |

Here are the default parameters for the `OpenAIJsonModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `gpt-3.5-turbo-1106` | The name of the model to use. |
| `num_call` | `int` | 1 | The number of calls to make to the OpenAI model |
| `temperature` | `float` | 1.5 | The temperature to use for the OpenAI model. |
| `response_format` | `Dict[str, str]` | {"type": "json_object"} | The response format to use for the OpenAI model. |

See the [openai_json_model.ipynb notebook](./openai_json_model.ipynb) for a working example.

## HuggingfaceConfig
The `HuggingfaceConfig` configuration has the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [HuggingfaceModelFlow](../../README.md#model) | The name of the flow to run. |
| `few_shot_template` | `List[Dict[str, str]]` | {} | The template to use for the few-shot model. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `HuggingfaceModelConfig` | The model configuration to use. |

Here are the default parameters for the `HuggingfaceModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `mistralai/Mistral-7B-Instruct-v0.1` | The name of the model to use. |

See the [huggingface_model.ipynb notebook](./huggingface_model.ipynb) for a working example.

> *Note: To run this config, you must also install the `transformers`, `torch`, `bitsandbytes`, and `accelerate` libraries, as detailed in the [Installation](../../README.md#installation) section.*

## LMQGModelConfig
The `LMQGModelConfig` configuration runs with the following default parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flow_name` | `str` | [LMQGModelFlow](../../README.md#model) | The name of the flow to run. |
| `few_shot_template` | `List[Dict[str, str]]` | {} | The template to use for the few-shot model. |
| `num_threads` | `int` | 1 | The number of threads to use. |
| `model_config` | `ModelConfig` | `LMQGModelConfig` | The model configuration to use. |

Here are the default parameters for the `LMQGModelConfig`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `lmqg/t5-base-squad-qg-ae` | The name of the model to use. |

See the [lmqg_model.ipynb notebook](./lmqg_model.ipynb) for a working example.
