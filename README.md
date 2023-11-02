# 🌊 uniflow
<p align="center">
  <a href="/LICENSE"><img alt="License Apache-2.0" src="https://img.shields.io/github/license/cambioml/uniflow?style=flat-square"></a>
  <a href="https://pypi.org/project/uniflow"><img src="https://img.shields.io/pypi/v/uniflow.svg" alt="pypi_status" /></a>
  <a href="https://github.com/cambioml/uniflow/graphs/commit-activity"><img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/cambioml/uniflow?style=flat-square"/></a>
  <a href="https://join.slack.com/t/cambiomlworkspace/shared_invite/zt-1zes33rmt-20Rag043uvExUaUdvt5_xQ"><img src="https://badgen.net/badge/Join/Community/cyan?icon=slack" alt="Slack" /></a>
</p>

`uniflow` is a unified interface for synthetic data generation. You can generate and augment synthetic data from raw text or other unstructured data using one of or multiple `uniflow` flows, including flows to [augment structured data](#augment-structured-data), [generate structured data from unstructured text](#generate-structured-data-from-unstructured-text), and [generate structured data from unstructured text (self instructed)](#generate-structured-data-from-unstructured-text-self-instructed).

Built by [CambioML](https://www.cambioml.com/).

## Quick Install

```
pip3 install uniflow
```

See more details at the [full installation](#installation).

## Overview
For all the flows, you must first import the Client interface and the `uniflow` constants.

```
from uniflow.client import Client
from uniflow.flow.constants import (OUTPUT_NAME, QAPAIR_DF_KEY, ...)
```

Then you can create a `Client` object to run the a particular flow.
```
client = Client(YOUR_FLOW_KEY)
```

Here is a table of the different flows and their corresponding keys, and input file types.
| Flow | Key | Input File Type |
| ------------- | ------------- | ------------- |
| [Augment Structured Data](#augment-structured-data) | flow_data_gen  | .csv |
| [Generate Structured Data from Unstructured Text](#generate-structured-data-from-unstructured-text) | flow_data_gen_text | .txt, .html |
| [Generate and Augment Structured Data from Unstructured Text](#generate-and-augment-structured-data-from-unstructured-text) | flow_text_plus_data_gen | .txt, .html |
| [Generate Structured Data from Unstructured Text (Self Instructed)](#generate-structured-data-from-unstructured-text-self-instructed) | flow_self_instructed_gen | .html |

Every flow takes a list of input dictionaries. Each dictionary has its own input file, with the `INPUT_FILE` key as shown below:
```
from uniflow.flow.constants import INPUT_FILE
input_dict = {INPUT_FILE: input_file}
```
The `input_file` is the full path to the input data file.

You can have multiple dictionaries in the input list, each with a different structured data file.
```
input_list = [input_dict1, input_dict2,...]
```

Next, you can use the `client` object to run the flow on the input list.
```
output_list = client.run(input_list)
```

The output list will have the same number of dictionaries as the input list, with each dictionary containing the corresponding generated QA pairs.

All of the flows will have the output dictionary with the following listed at the OUTPUT_NAME key. Within the output dictionary, you have the following keys and corresponding values
| Key | Description |
| ------------- | ------------- |
| `QAPAIR_DF_KEY` | The output QA dataframe |
| `OUTPUT_FILE` | The output file path |
| `ERROR_LIST` (optional) | List of any errors |

Here's an example of how to access the output QA dataframe from the first output dictionary in the output list.

```
output_dict1 = output_list[0]
output_dict1[OUTPUT_NAME][0][QAPAIR_DF_KEY] #this will print the output QA dataframe
```
### Examples
For more examples, check out the [QA Generation](example/qa_generation) and [Self-Instructed](example/self_instructed_ft) folders.

## Flows
`uniflow` lets you easily generate synthetic data from raw text (including `.txt`, `.html`, `.pdf`, etc.). Here are the flows for common applications:

### Augment Structured Data
Given existing structured data (e.g. sample Question-Answer (QA) pairs), augment more QA pairs using the `Client("flow-data-gen")` interface.

#### Example
Check out this [example](example/qa_generation/README.md#augment-structured-data) and this [notebook](example/qa_generation/data_generation.ipynb) to get started.

### Generate Structured Data from Unstructured Text
Generate structured data (e.g. Question-Answer pairs) from unstructured text using the `Client("flow-data-gen-text")` interface.

#### Example
Check out this [example](example/qa_generation/README.md#generate-structured-data-from-unstructured-text) and this [notebook](example/qa_generation/data_generation_text.ipynb) to get started.

### Generate and Augment Structured Data from Unstructured Text
Using the `Client("flow_text_plus_data_gen")` interface, you can run the previous two flows in sequence to generate structured data from unstructured text, and then augment more data from the structured data.

#### Example
Check out this [example](example/qa_generation/README.md#generate-and-augment-structured-data-from-unstructured-text) and this [notebook](example/qa_generation/text_plus_data_generation.ipynb) to get started.

### Generate Structured Data from Unstructured Text (Self Instructed)
Generate data from unstructured text using the `Client("flow_self_instructed_gen")` interface. This flow generates question answer pairs from unstructured `.html` files.

#### Example
Check out this [example](example/self_instructed_ft/README.md) and this [notebook](example/self_instructed_ft/demo_self_instruct_ft.ipynb) to get started.

## Installation
To get started with `uniflow`, you can install it using `pip` in a `conda` environment.

First, create a conda environment on your terminal using:
```
conda create -n uniflow python=3.10 -y
conda activate uniflow  # some OS requires `source activate uniflow`
```

Then install `flow` and the compatible [pytorch based on your OS](https://pytorch.org/get-started/locally/):
```
pip3 install uniflow
pip3 install torch
```

Finally, if you are on a GPU, install [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
```

Congrats you have finished the installation!

## Dev Setup
If you are interested in contributing to us, here are the preliminary development setups.

### API keys
If you are running one of the following flows, you will have to set up your OpenAI API key.
- flow_data_gen
- flow_text_plus_data_gen

To do so, create a `.env` file in your root uniflow folder. Then add the following line to the `.env` file:
```
OPENAI_API_KEY=YOUR_API_KEY
```
### Backend Dev Setup

```
conda create -n uniflow python=3.10
conda activate uniflow
cd uniflow
pip3 install poetry --no-root
```

### EC2 Dev Setup
If you are on EC2, you can launch a GPU instance with the following config:
- EC2 `g4dn.xlarge` (if you want to run a pretrained LLM with 7B parameters)
- Deep Learning AMI PyTorch GPU 2.0.1 (Ubuntu 20.04)
    <img src="example/image/readme_ec2_ami.jpg" alt="Alt text" width="50%" height="50%"/>
- EBS: at least 100G
    <img src="example/image/readme_ec2_storage.png" alt="Alt text" width="50%" height="50%"/>