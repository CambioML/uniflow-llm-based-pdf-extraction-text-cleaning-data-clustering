# 🌊 uniflow
<p align="center">
  <a href="/LICENSE"><img alt="License Apache-2.0" src="https://img.shields.io/github/license/cambioml/uniflow?style=flat-square"></a>
  <a href="https://pypi.org/project/uniflow"><img src="https://img.shields.io/pypi/v/uniflow.svg" alt="pypi_status" /></a>
  <a href="https://github.com/cambioml/uniflow/graphs/commit-activity"><img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/cambioml/uniflow?style=flat-square"/></a>
  <a href="https://join.slack.com/t/cambiomlworkspace/shared_invite/zt-1zes33rmt-20Rag043uvExUaUdvt5_xQ"><img src="https://badgen.net/badge/Join/Community/cyan?icon=slack" alt="Slack" /></a>
</p>

`uniflow` is a unified interface to solve data augmentation problem for LLM training.

Built by [CambioML](https://www.cambioml.com/).

## Quick Install

```
pip3 install uniflow
```

See more details at the [full installation](#installation).

## Overview
For all the flows, you must first import the `Client` interface and the specific `uniflow` configuration that you want to use.

```
from uniflow.client import Client
from uniflow.config import UNIFLOW_CONFIG
```

Then you can create a `Client` object to run the a particular flow.
```
client = Client(UNIFLOW_CONFIG)
```

Here is a table of the different pre-defined configurations you can use and their corresponding Flows, Model Configs, LLMs:
| Config | [Flow](#flows) | [Model Config](#model-config) | LLM |
| ------------- | ------------- | ------------- | ------------- |
| __Config__ | `ModelFLow` | `ModelConfig` | `gpt-3.5-turbo-1106`|
| __OpenAIConfig__ | `ModelFLow` | `OpenAIModelConfig` | `gpt-3.5-turbo-1106`|
| [__OpenAIJsonConfig__](./example/model/README.md#openaijsonconfig) | `OpenAIJsonModelFlow` | `ModelConfig` | `gpt-3.5-turbo-1106` |
| [__OpenAIFewShotConfig__](./example/model/README.md#openaifewshotconfig) | `FewShotModelFlow` | `OpenAIModelConfig` |  `gpt-3.5-turbo-1106` |
| [__HuggingfaceConfig__](./example/model/README.md#huggigfacemodelconfig) | `HuggingFaceModelFlow` | `HuggingfaceModelConfig` |  `mistralai/Mistral-7B-Instruct-v0.1` |
| [__LMQGConfig__](./example/model/README.md#lmqgmodelconfig) | `LMQGModelFlow` | `LMQGModelConfig` | `lmqg/t5-base-squad-qg-ae` |

You can also run your own custom flows by passing in a custom configurations and/or parameters to the `Client` object. See the [Custom Configuration](#custom-configuration) section for more details.

Every flow takes a list of dictionaries as input. Each dictionary in the input list should have data to prompt in a format similar one of the following:

1. A simple list of `context` dicts, which can be used with a few-shot model:
      ```
      data = [
    {"context": """It was a sunny day and the sky color is blue.""",},
    {"context": """My name is bobby and I am a talent software engineer working on AI/ML.""",}
    ]
      ```
    This will be appended to the `few_shot_template` in the configuration. See the [few_shot_model.ipynb notebook](./few_shot_model.ipynb) for a working example.

1. A more complex list of dictionaries with a single `instruction` and an `examples` list:
    ```
    data = [{
      "instruction": """Generate one question and its corresponding answer based on the context. Following the format of the examples below to include context, question, and answer in the response.""",
      "examples": [
          {
              "context": """The quick brown fox jumps over the lazy dog.""",
              "question": """What is the color of the fox?""",
              "answer": """brown."""
          },
          {
              "context": """The quick brown fox jumps over the lazy black dog.""",
              "question": """What is the color of the dog?""",
              "answer": """black."""
          },
          {
              "context": """It was a sunny day and the sky color is blue.""",
              "question": """""",
              "answer": """""",
          }
        ],
    },
    ...]
    ```

Next, use the `client` object to run the flow on the input data.
```
output = client.run(data)
```

For example if you want to run the `FewShotModelFlow` flow with the `OpenAIFewShotConfig` configuration, you can do the following:
```
from uniflow.client import Client
from uniflow.config import OpenAIFewShotConfig

data = [
    {"context": """It was a sunny day and the sky color is blue.""",},
    {"context": """My name is bobby and I am a talent software engineer working on AI/ML.""",}
    ]

config = OpenAIFewShotConfig()
client = Client(config)

output = client.run(data)

output[0]['output'][0]
```

You will see output similar to the following:
```
{'response': [{'context': 'It was a sunny day and the sky color is blue.',
   'question': 'what color is the sky on a sunny day?',
   'answer': 'blue.'}],
 'error': 'Failed to deserialize 0 examples'}
 ```
## Flows
The `Flow` determines which `uniflow Model` is run. All the different flows are built on the `Flow` class, which converts the input data into nodes. The `Model` determines how the input data is serialized and deserialized.

Here is a table of the different `Flows` and their corresponding `Models`:
| Flow | Model |
| ------------- | ------------- |
| `ModelFlow` | [`JsonModel`](#jsonmodel) |
| `FewShotModelFlow` | [`FewShotModel`](#fewshotmodel) |
| `OpenAIJsonModelFlow` | [`OpenAIJsonModel`](#openaijsonmodel) |
| `HuggingFaceModelFlow` | [`Model`](#model) |
| `LMQGModelFlow` | [`Model`](#model) |

### Model
#### Serialization
This model serializes the input data into a single string, with the format:
```
KEY0: VALUE0\n
KEY1: VALUE1\n
...
```
#### Deserialization
It returns the raw data in the following object:
```
{
    RESPONSE: data,
    ERROR: "Failed to deserialize 0 examples",
}
```
### JsonModel
#### Serialization
This model first appends the `data` to the `few_shot_template` in the configuration. It then serializes the data using `json.dumps`.
#### Deserialization
It iterates through the data and parses the output data into a `output_list`. Then returns the `output_list` in the following object:
```
{
    RESPONSE: output_list,
    ERROR: "Failed to deserialize 0 examples",
}
```

### FewShotModel
#### Serialization
The `FewShotModel` serializes the input data into a few more steps. First, it adds the input `data` to the `examples` list within the `few_shot_template`. Then it serializes the data into a single string, with the format:
```
KEY0: VALUE0\n
KEY1: VALUE1\n
...
```
#### Deserialization
It first runs through a data filtering function to parse the output data into a `output_list`. Then returns the `output_list` in the following object:
```
{
    RESPONSE: output_list,
    ERROR: "Failed to deserialize 0 examples",
}
```
### OpenAIJsonModel
#### Serialization
Same as the [`Model`](#serialization).

#### Deserialization
Same as from the [`Model`](#deserialization).

## Model Config
The __Model Config__ is a configuration that is passed to the base `Config` object and determines which LLM model is used. It also has parameters that are specific to the LLM model.

### ModelConfig
The base config is called `ModelConfig` and has the following parameters:
 Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `model_name` | str | gpt-3.5-turbo-1106 | name of the LLM model used |

### OpenAIModelConfig
The `OpenAIModelConfig` inherits from the `ModelConfig` and has the following additional parameters:
| Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `num_calls` | int | 1 | The number of calls to make to the OpenAI API. |
| `temperature` | float | 1.5 | The temperature to use for the OpenAI API. |
| `response_format` | Dict[str, str] | {"type": "text"} | The response format to use for the OpenAI API. |

### OpenAIJsonModelConfig
The `OpenAIJsonModelConfig` inherits from the `ModelConfig` and has the following additional parameters:
| Parameter | Type |  Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `num_calls` | int | 1 | The number of calls to make to the OpenAI API. |
| `temperature` | float | 1.5 | The temperature to use for the OpenAI API. |
| `response_format` | Dict[str, str] | {"type": "json_object"} | The response format to use for the OpenAI API. |

### HuggingfaceModelConfig
The `HuggingfaceModelConfig` inherits from the `ModelConfig`, but overrides the `model_name` parameter to use the `mistralai/Mistral-7B-Instruct-v0.1` model by default.
| Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `model_name` | str | mistralai/Mistral-7B-Instruct-v0.1 | name of the LLM model used |

### LMQGModelConfig
The `LMQGModelConfig` inherits from the `ModelConfig`, but overrides the `model_name` parameter to use the `lmqg/t5-base-squad-qg-ae` model by default.

| Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `model_name` | str | lmqg/t5-base-squad-qg-ae | name of the LLM model used |

## Few Shot Template
The `OpenAIFewShotConfig` is the only configuration that has a default `feww_shot_template`. The default template is the following:
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

## Custom Configuration
You can also configure the flows by passing custom configurations or arguments to the `Client` object.

Every configuration inherits from the `Config` configuration, which has the following parameters:
| Parameter | Type | Description |
| ------------- | ------------- | ------------- |
| `flow_name` | str | The name of the flow to run. |
| `few_shot_template` | Dict[str, str] | The template to use for the `FewShotModelFlow`. |
| `num_threads` | int | The number of threads to use for the flow. |
| `model_config` | `ModelConfig` | The configuration to pass to the model. |

You can further configure the `model_config` by passing in one of the `Model Configs` with custom parameters.

Here is an example of how to pass in a custom configuration to the `Client` object:
```
from uniflow.client import Client
from uniflow.config import OpenAIFewShotConfig
from uniflow.model.config import OpenAIModelConfig

data = [
    {"context": """It was a sunny day and the sky color is blue.""",},
    {"context": """My name is bobby and I am a talent software engineer working on AI/ML.""",}
    ]

config = OpenAIFewShotConfig(
  num_threads=2,
  model_config=OpenAIModelConfig(
    model_name="gpt-4",
    num_calls=2,
    temperature=0.5,
  ),
)
client = Client(config)
output = client.run(data)
```

As you can see, we are passing in a custom parameters to the `OpenAIModelConfig` to the `OpenAIFewShotConfig` configurations according to our needs.


## Examples
 For more examples, see the [example](./example/) folder.


## Installation
To get started with `uniflow`, you can install it using `pip` in a `conda` environment.

First, create a conda environment on your terminal using:
```
conda create -n uniflow python=3.10 -y
conda activate uniflow  # some OS requires `source activate uniflow`
```

Next, install the compatible pytorch based on your OS. 
- If you are on a GPU, install [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
    ```
    pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
    ```
- If you are on a CPU instance,
    ```
    pip3 install torch
    ```

Then, install `uniflow`:
```
pip3 install uniflow
```

Finally, if you are running the `HuggingfaceModelFlow`, you will also need to install the `transformers`, `accelerate`, `bitsandbytes`, `scipy` libraries:
```
pip3 install transformers accelerate bitsandbytes scipy
```

Congrats you have finished the installation!

## Dev Setup
If you are interested in contributing to us, here are the preliminary development setups.

### API keys
If you are running one of the following `OpenAI` flows, you will have to set up your OpenAI API key.

To do so, create a `.env` file in your root uniflow folder. Then add the following line to the `.env` file:
```
OPENAI_API_KEY=YOUR_API_KEY
```
### Backend Dev Setup

```
conda create -n uniflow python=3.10
conda activate uniflow
cd uniflow
pip3 install poetry
poetry install --no-root
```

### EC2 Dev Setup
If you are on EC2, you can launch a GPU instance with the following config:
- EC2 `g4dn.xlarge` (if you want to run a pretrained LLM with 7B parameters)
- Deep Learning AMI PyTorch GPU 2.0.1 (Ubuntu 20.04)
    <img src="example/image/readme_ec2_ami.jpg" alt="Alt text" width="50%" height="50%"/>
- EBS: at least 100G
    <img src="example/image/readme_ec2_storage.png" alt="Alt text" width="50%" height="50%"/>