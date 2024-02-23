# 🌊 uniflow
<p align="center">
  <a href="/LICENSE"><img alt="License Apache-2.0" src="https://img.shields.io/github/license/cambioml/uniflow?style=flat-square"></a>
  <a href="https://pypi.org/project/uniflow"><img src="https://img.shields.io/pypi/v/uniflow.svg" alt="pypi_status" /></a>
  <a href="https://github.com/cambioml/uniflow/graphs/commit-activity"><img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/cambioml/uniflow?style=flat-square"/></a>
  <a href="https://join.slack.com/t/cambiomlworkspace/shared_invite/zt-1zes33rmt-20Rag043uvExUaUdvt5_xQ"><img src="https://badgen.net/badge/Join/Community/cyan?icon=slack" alt="Slack" /></a>
</p>

`uniflow` provides a unified LLM interface to extract and transform and raw documents.
- Document types: Uniflow enables data extraction from [PDFs](https://github.com/CambioML/uniflow-llm-based-text-extraction-data-cleaning-clustering/blob/main/example/extract/extract_pdf_with_recursive_splitter.ipynb), [HTMLs](https://github.com/CambioML/uniflow-llm-based-text-extraction-data-cleaning-clustering/blob/main/example/extract/extract_html.ipynb) and [TXTs](https://github.com/CambioML/uniflow-llm-based-text-extraction-data-cleaning-clustering/blob/main/example/extract/extract_txt.ipynb).
- LLM agnostic: Uniflow supports most common-used LLMs for text tranformation, including
    - OpenAI models ([GPT3.5 and GPT4](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/main/example/transform/openai_pdf_source_10k_summary.ipynb)), 
    - Google Gemini models ([Gemini 1.5](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/main/example/transform/google_model.ipynb), [MultiModal](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/main/example/transform/google_multimodal_model.ipynb)), 
    - AWS [BedRock](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/main/example/rater/bedrock_classification.ipynb) models, 
    - Huggingface open source models including [Mistral-7B](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/0222/example/transform/huggingface_model_5QAs.ipynb), 
    - Azure OpenAI models, etc.


## :question: The Problems to Tackle

Uniflow addresses two key challenges in preparing LLM training data for ML scientists: 
- first, extracting legacy documents like PDFs and Word files into clean text, which LLMs can learn from, is tricky due to complex PDF layouts and missing information during extraction; and 
- second, the labor-intensive process of transforming extracted data into a format suitable for training LLMs, which involves creating datasets with both preferred and rejected answers for each question to support feedback-based learning techniques.

Hence, we built Uniflow, a unified LLM interface to extract and transform and raw documents.

## :seedling: Use Cases

Uniflow aims to help every data scientist generate their own privacy-perserved, ready-to-use training datasets for LLM finetuning, and hence make finetuning LLMs more accessible to everyone:rocket:. 

Check Uniflow hands-on solutions:

- [Extract financial reports (PDFs) into summerrization](https://github.com/CambioML/cambio-cookbook/blob/main/examples/10K_Evaluator/10K_PDF_Summary.ipynb)
- [Extract financial reports (PDFs) and finetune financial LLMs](https://github.com/CambioML/cambio-cookbook/blob/main/examples/10K_Evaluator/10K_PDF_Evaluator.ipynb)
- [Extract A Math Book (HTMLs) into your question answer dataset](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/main/example/transform/self_instruct_custom_html_source.ipynb)
- [Extract PDFs into your question answer dataset](https://github.com/CambioML/uniflow-llm-based-pdf-extraction-text-cleaning-data-clustering/blob/main/example/transform/huggingface_model_5QAs.ipynb)
- Build RLHF/RLAIF perference datasets for LLM finetuning

---

## :computer: Installation

Installing `uniflow` takes about 5-10 minutes if you follow the 3 steps below:

1. Create a conda environment on your terminal using:
    ```
    conda create -n uniflow python=3.10 -y
    conda activate uniflow  # some OS requires `source activate uniflow`
    ```

2. Install the compatible pytorch based on your OS.
    - If you are on a GPU, install [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
        ```
        pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
        ```
    - If you are on a CPU instance,
        ```
        pip3 install torch
        ```

3. Install `uniflow`:
    ```
    pip3 install uniflow
    ```
    - (Optional) If you are running one of the following `OpenAI` flows, you will have to set up your OpenAI API key. To do so, create a `.env` file in your root uniflow folder. Then add the following line to the `.env` file:
        ```
        OPENAI_API_KEY=YOUR_API_KEY
        ```

    - (Optional) If you are running the `HuggingfaceModelFlow`, you will also need to install the `transformers`, `accelerate`, `bitsandbytes`, `scipy` libraries:
        ```
        pip3 install transformers accelerate bitsandbytes scipy
        ```
    - (Optional) If you are running the `LMQGModelFlow`, you will also need to install the `lmqg` and `spacy` libraries:
        ```
        pip3 install lmqg spacy
        ```

Congrats you have finished the installation!


## :man_technologist: Dev Setup
If you are interested in contributing to us, here are the preliminary development setups.

```
conda create -n uniflow python=3.10 -y
conda activate uniflow
cd uniflow
pip3 install poetry
poetry install --no-root
```

### AWS EC2 Dev Setup
If you are on EC2, you can launch a GPU instance with the following config:
- EC2 `g4dn.xlarge` (if you want to run a pretrained LLM with 7B parameters)
- Deep Learning AMI PyTorch GPU 2.0.1 (Ubuntu 20.04)
    <img src="example/image/readme_ec2_ami.jpg" alt="Alt text" width="50%" height="50%"/>
- EBS: at least 100G
    <img src="example/image/readme_ec2_storage.png" alt="Alt text" width="50%" height="50%"/>

### API keys
If you are running one of the following `OpenAI` flows, you will have to set up your OpenAI API key.

To do so, create a `.env` file in your root uniflow folder. Then add the following line to the `.env` file:
```
OPENAI_API_KEY=YOUR_API_KEY
```

---

# :scroll: Uniflow Manual

## Overview
To use `uniflow`, follow of three main steps:
1. **Pick a [`Config`](#config)**\
    This determines the LLM and the different configurable parameters.

1. **Construct your [`Prompts`](#prompting)**\
    Construct the context that you want to use to prompt your model. You can configure custom instructions and examples using the [`PromptTemplate`](#PromptTemplate) class.

1. **Run your [`Flow`](#running-the-flow)**\
    Run the flow on your input data and generate output from your LLM.

> *Note: We're currently building have `Preprocessing` flows as well to help process data from different sources, such as `pdf`, `html`, `Markdown`, and more.*

## 1. Config
The `Config` determines which LLM is used and how the input data is serialized and deserialized. It also has parameters that are specific to the LLM.

Here is a table of the different pre-defined configurations you can use and their corresponding LLMs:
| Config | LLM |
| ------------- | ------------- |
| __Config__ | [`gpt-3.5-turbo-1106`](https://platform.openai.com/docs/models/gpt-3-5) |
| __OpenAIConfig__ | [`gpt-3.5-turbo-1106`](https://platform.openai.com/docs/models/gpt-3-5)|
| __HuggingfaceConfig__| [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |
| __LMQGConfig__ | [`lmqg/t5-base-squad-qg-ae`](https://huggingface.co/lmqg/t5-base-squad-qg-ae) |

You can run each config with the defaults, or you can pass in custom parameters, such as `temperature` or `batch_size` to the config for your use case. See the [advanced custom configuration](#advanced-custom-configuration) section for more details.

## 2. Prompting
By default, `uniflow` is set up to generate questions and answers based on the `Context` you pass in. To do so, it has a default instruction and few-shot examples that it uses to guide the LLM.

Here is the default instruction:
```
Generate one question and its corresponding answer based on the last context in the last example. Follow the format of the examples below to include context, question, and answer in the response
```

Here are the default few-shot examples:
```
    context="The quick brown fox jumps over the lazy brown dog.",
    question="What is the color of the fox?",
    answer="brown."

    context="The quick brown fox jumps over the lazy black dog.",
    question="What is the color of the dog?",
    answer="black."
```

To run with these default instructions and examples, all you need to do is pass in a list of  `Context` objects to the flow. `uniflow` will then generate a custom prompt with the instructions and few-shot examples for each `Context` object to send to the LLM. See the [Running the flow](#running-the-flow) section for more details.

### Context
The `Context` class is used to pass in the context for the LLM prompt. A `Context` consists of a `context` property, which is a string of text.

To run `uniflow` with the default instructions and few-shot examples, you can pass in a list of `Context` objects to the flow. For example:
```
from uniflow.op.prompt import Context

data = [
    Context(
        context="The quick brown fox jumps over the lazy brown dog.",
    ),
    ...
]

client.run(data)
```

For a more detailed overview of running the flow, see the [Running the flow](#running-the-flow) section.

### PromptTemplate
If you want to run with a custom prompt instruction or few-shot examples, you can use the `PromptTemplate` object. It has `instruction` and `example` properties.

| Property | Type | Description |
| ------------- | ------------- | ------------- |
| `instruction` | str | Detailed instructions for the LLM |
| `examples` | List[Context] | The few-shot examples. |

You can overwrite any of the defaults as needed.

To see an example of how to use the `PromptTemplate` to run `uniflow` with a custom `instruction`, few-shot examples, and custom `Context` fields to generate a summary, check out the [openai_pdf_source_10k_summary notebook](./example/model/openai_pdf_source_10k_summary.ipynb)


## Running the Flow
Once you've decided on your `Config` and prompting strategy, you can run the flow on the input data.

1. Import the `uniflow` `Client`, `Config`, and `Context` objects.
    ```
    from uniflow.flow.client import TransformClient
    from uniflow.flow.config import TransformOpenAIConfig, OpenAIModelConfig
    from uniflow.op.prompt import Context
    ```
1. Preprocess your data in to chunks to pass into the flow. In the future we will have `Preprocessing` flows to help with this step, but for now you can use a library of your choice, like [pypdf](https://pypi.org/project/pypdf/), to chunk your data.
    ```
    raw_input_context = ["It was a sunny day and the sky color is blue.", "My name is bobby and I am a talent software engineer working on AI/ML."]
    ```

1. Create a list of `Context` objects to pass your data into the flow.
    ```
    data = [
        Context(context=c)
        for c in raw_input_context
    ]
    ```

1. [Optional] If you want to use a customized instruction and/or examples, create a `PromptTemplate`.
    ```
    from uniflow.op.prompt import PromptTemplate

    guided_prompt = PromptTemplate(
    instruction="Generate a one sentence summary based on the last context below. Follow the format of the examples below to include context and summary in the response",
    few_shot_prompt=[
        Context(
            context="When you're operating on the maker's schedule, meetings are a disaster. A single meeting can blow a whole afternoon, by breaking it into two pieces each too small to do anything hard in. Plus you have to remember to go to the meeting. That's no problem for someone on the manager's schedule. There's always something coming on the next hour; the only question is what. But when someone on the maker's schedule has a meeting, they have to think about it.",
            summary="Meetings disrupt the productivity of those following a maker's schedule, dividing their time into impractical segments, while those on a manager's schedule are accustomed to a continuous flow of tasks.",
        ),
    ],
    )
    ```

1. Create a `Config` object to pass into the `Client` object.
    ```
    config = TransformOpenAIConfig(
        prompt_template=guided_prompt,
        model_config=OpenAIModelConfig(
            response_format={"type": "json_object"}
        ),
    )
    client = TransformClient(config)
    ```

1. Use the `client` object to run the flow on the input data.

    ```
    output = client.run(data)
    ```

1. Process the output data. By default, the LLM output will be a list of output dicts, one for each `Context` passed into the flow. Each dict has a `response` property which has the LLM response, as well as any errors. For example `output[0]['output'][0]` would look like this:
    ```
    {
        'response': [{'context': 'It was a sunny day and the sky color is blue.',
        'question': 'What was the color of the sky?',
        'answer': 'blue.'}],
        'error': 'No errors.'
    }
    ```

## Examples
 For more examples, see the [example](./example/model) folder.

## Advanced Custom Configuration
You can also configure the flows by passing custom configurations or arguments to the `Config` object if you want to further tune specific parameters like the the LLM model, number of threads, the temperature, and more.

Every configuration has the following parameters:
| Parameter | Type | Description |
| ------------- | ------------- | ------------- |
| `prompt_template` | `PromptTemplate` | The template to use for the guided prompt. |
| `num_threads` | int | The number of threads to use for the flow. |
| `model_config` | `ModelConfig` | The configuration to pass to the model. |

You can further configure the `model_config` by passing in one of the `Model Configs` with custom parameters.

### Model Config
The __Model Config__ is a configuration that is passed to the base `Config` object and determines which LLM model is used and has parameters that are specific to the LLM model.

#### ModelConfig
The base config is called `ModelConfig` and has the following parameters:
 Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `model_name` | str | gpt-3.5-turbo-1106 | [OpenAI site](https://platform.openai.com/docs/models/gpt-3-5) |

#### OpenAIModelConfig
The `OpenAIModelConfig` inherits from the `ModelConfig` and has the following additional parameters:
| Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `num_calls` | int | 1 | The number of calls to make to the OpenAI API. |
| `temperature` | float | 1.5 | The temperature to use for the OpenAI API. |
| `response_format` | Dict[str, str] | {"type": "text"} | The response format to use for the OpenAI API. Can be "text" or "json" |

#### HuggingfaceModelConfig
The `HuggingfaceModelConfig` inherits from the `ModelConfig`, but overrides the `model_name` parameter to use the `mistralai/Mistral-7B-Instruct-v0.1` model by default.
| Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `model_name` | str | mistralai/Mistral-7B-Instruct-v0.1 | [Hugging Face site](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |
| `batch_size` | int | 1 | The batch size to use for the Hugging Face API. |

#### LMQGModelConfig
The `LMQGModelConfig` inherits from the `ModelConfig`, but overrides the `model_name` parameter to use the `lmqg/t5-base-squad-qg-ae` model by default.

| Parameter | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `model_name` | str | lmqg/t5-base-squad-qg-ae | [Hugging Face site](https://huggingface.co/lmqg/t5-base-squad-qg-ae) |
| `batch_size` | int | 1 | The batch size to use for the LMQG API. |

### Custom Configuration Example
Here is an example of how to pass in a custom configuration to the `Client` object:
```
from uniflow.flow.client import TransformClient
from uniflow.flow.config import TransformOpenAIConfig, OpenAIModelConfig
from uniflow.op.prompt import Context


contexts = ["It was a sunny day and the sky color is blue.", "My name is bobby and I am a talent software engineer working on AI/ML."]

data = [
    Context(
        context=c
    )
    for c in contexts
]

config = OpenAIConfig(
  num_threads=2,
  model_config=OpenAIModelConfig(
    model_name="gpt-4",
    num_calls=2,
    temperature=0.5,
  ),
)
client = TransformClient(config)
output = client.run(data)
```

As you can see, we are passing in a custom parameters to the `OpenAIModelConfig` to the `OpenAIConfig` configurations according to our needs.