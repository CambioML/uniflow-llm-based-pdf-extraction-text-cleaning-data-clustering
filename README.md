# 🌊 🔀 uniflow
<p align="center">
  <a href="/LICENSE"><img alt="License Apache-2.0" src="https://img.shields.io/github/license/cambioml/uniflow?style=flat-square"></a>
  <a href="https://pypi.org/project/uniflow"><img src="https://img.shields.io/pypi/v/uniflow.svg" alt="pypi_status" /></a>
  <a href="https://github.com/cambioml/uniflow/graphs/commit-activity"><img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/cambioml/uniflow?style=flat-square"/></a>
  <a href="https://join.slack.com/t/cambiomlworkspace/shared_invite/zt-1zes33rmt-20Rag043uvExUaUdvt5_xQ"><img src="https://badgen.net/badge/Join/Community/cyan?icon=slack" alt="Slack" /></a>
</p>

`uniflow` is a unified interface for synthetic data generation. You can generate and augment synthetic data from raw text or data using one of or multiple `uniflow` flows, including `DataGenFlow`, `DataGenTextFlow`, `TextPlusDataGenFlow`, and `SelfInstructedGenFlow`.

Built by [CambioML](https://www.cambioml.com/).

## Quick Install

```
pip3 install uniflow
```

See more details at the [full installation](https://github.com/CambioML/uniflow/tree/main#Installation).


## Features
`uniflow` lets you easily generate synthetic data from raw text (including `.txt`, `.html`, `.pdf`, etc.). Here are the flows for common applications:

### Augment Structured Data
Given existing structured sample data (e.g. Question-Answer (QA) pairs), augment more QA pairs using the `DataGenFlow` interface.

#### Example
First, load a raw structured file (e.g. `.csv`) with Question and Answer columns.
```
# Initiate flow
flow = DataGenFlow()
# Load data
qaa = pd.read_csv(f"{YOUR_CSV_FILE}", encoding = "utf8")`
```

Then you can generate more data from the original structured data.
```
input_dict = {"qaa": qaa[:]}
output_dict = flow(input_dict)

```
### Generate Structured Data from Unstructured Text
Generate structured data (e.g. Question-Answer pairs) from unstructured text using the `DataGenTextFlow` interface.

#### Example
First, load a text file (e.g. `.txt`) from which you would like to generate the Question-Answer pairs.
```
with open(os.path.join(f"{YOUR_FILE_PATH}", "YOUR_FILE.txt"), "r") as file:
    context = file.read()
```
Then you can generate the Question-Answer pairs from the text.
```
flow = DataGenTextFlow()
input_dict = {"context": context}
output_dict = flow(input_dict)
```

### Generate and Augment Structured Data from Unstructured Text
Using the `TextPlusDataGenFlow`, you can run the previous two flows in sequence to generate structured data from unstructured text, and then augment more data from the structured data.

#### Example
First, load a raw text file (e.g. `.txt`) from which you would like to generate the structured data (e.g. Question-Answer pairs).
```
with open(os.path.join(f"{YOUR_FILE_PATH}", "YOUR_FILE.txt"), "r") as file:
    context = file.read()
```
Then you can generate the Question-Answer pairs from the raw text.
```
flow = TextPlusDataGenFlow()
input_dict = {"context": context}
output_dict = flow(input_dict)
```

### Generate Structured Data from Unstructured Text (Self Instructed)
Generate data from unstructured text using the `SelfInstructedGenFlow` interface.

First, place your raw text file (e.g. `.html`) in the same directory as your python file.

Then you can generate the structured data (e.g. Question-Answer pairs) from the raw text.
```
# Initiate flow
flow = SelfInstructedGenFlow()
input_dict = {constants.HTML_KEY: YOUR_HTML_FILE_PATH}

# Run flow
output_dict = flow(input_dict)
```

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