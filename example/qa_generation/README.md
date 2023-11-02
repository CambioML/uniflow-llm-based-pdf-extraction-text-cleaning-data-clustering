# Question-Answer Data Generation
`uniflow` gives you the ability to generate synthetic data from raw text (including `.txt` and `.html`), as well as augment your structured data with more structured data (e.g. from a `.csv`).

## Augment Structured Data
The [`data_generation` notebook](data_generation.ipynb) shows how to use `uniflow` to generate more QA pairs from existing QA pairs.

It loads the `qd_immigration_short.csv` file,  which has just the first 3 questions and answers, and generates more QA pairs from them.

Then it runs on the full `qd_immigration.csv` file, which has 96 questions and answers, and generates more QA pairs from them. This will take significantly longer than the first example.

### Setup
To get started, import Client and the input keys from `uniflow`:
```
from uniflow.client import Client
from uniflow.flow.constants import (OUTPUT_NAME, QAPAIR_DF_KEY)
```

Then, initiate a `Client` flow for `flow_data_gen`:
```
client = Client("flow_data_gen")
```

### OpenAI API Key
This flow uses OpenAI. You will need to set up your environment with your OpenAI API key. [See this example.](../../README.md#api-keys)

### QA Factor
The number of new questions generated is determined by the QA_FACTOR. For example, if there are 3 input QA pairs, and the QA_FACTOR is set to 3, then you should expect roughly 9 output QA pairs.

To set the QA_FACTOR, you can use the `config.set_qa_factor()` method to set the number of QA pairs you want to generate from each original QA pair. The default is 3.

```
import uniflow.config as config
config.set_qa_factor(5) # Set QA factor to 5 (default is 3)
```

### Input Format
This flow expects the `INPUT_FILE` key to be the key for the input dictionary. The value for this key should be the path to the `csv` file from which you want to generate the QA pairs. It must include question and answer columns with "Question" and "Answer" as the column names respectively. You can look at [qd_immigration.csv](data/raw_input/qd_immigration.csv) for an example of the input format.

```
input_dict = {INPUT_FILE: your_input_file_path}
```
### Process
Then you can run the flow on the input dictionaries:
```
input_list = [input_dict]
output_list = client.run(input_list)
```
The input QA pairs are run through an OpenAI model [text-davinci-003](https://platform.openai.com/docs/models/gpt-3-5).


### Output
The output Question-Answer pairs are saved to `csv` and `json` files called `output_qa_augment_data` in the `./data/output` folder.

## Generate Structured Data from Unstructured Text
You can also use `uniflow` to generate QA pairs from raw text. The [`data_generation_text` notebook](data_generation_text.ipynb) shows how to use `uniflow` to generate QA pairs from raw text.

It loads either a raw .txt file (`umich.txt`) or an html file (`22.4_multivariable-calculus.html`). It then uses the [lmqg](https://huggingface.co/lmqg) package to generate questions from the text by paragraph.

### Setup
To get started, import Client and the input keys from `uniflow`:
```
from uniflow.client import Client
from uniflow.flow.constants import (OUTPUT_NAME, INPUT_FILE)
```

Then, initiate a `Client` flow for `flow_data_gen_text`:
```
client = Client("flow_data_gen_text")
```

### Input Format
This flow expects the INPUT_FILE key to be the key for the input dictionary. The value for this key should be the path to the raw text file (e.g. `.txt` or `.html`) from which you want to generate the QA pairs.

```
input_dict = {INPUT_FILE: your_input_file_path}
```

### Process
Then you can run the flow on the input dictionaries:
```
input_list = [input_dict]
output_list = client.run(input_list)
```

The raw text is first split into paragraphs. Then, the [lmqg](https://huggingface.co/lmqg) package is used to generate question-answer pairs from each paragraph. The number of QA pairs is determined by [lmqg](https://huggingface.co/lmqg), and is not configurable.

### Output
The output Question-Answer pairs are saved to `output_qa_text_data.csv` file in the `./data/output` folder.

## Generate and Augment Structured Data from Unstructured Text
This flow combines the previous two flows to generate QA pairs from raw text, and then augment more QA pairs from the generated QA pairs. The [`text_plus_data_generation` notebook](text_plus_data_generation.ipynb) shows how to use `uniflow` in this way.

### Setup
To get started, import Client and the input keys from `uniflow`:
```
from uniflow.client import Client
from uniflow.flow.constants import (OUTPUT_NAME, INPUT_FILE)
```

Then, initiate a `Client` flow for `flow_text_plus_data_gen`:
```
client = Client("flow_text_plus_data_gen")
```

### OpenAI API Key
This flow uses OpenAI. You will need to set up your environment with your OpenAI API key. [See this example.](../../README.md#api-keys)

### QA Factor
You can set the [QA_FACTOR](#qa-factor), same as with the [Augment Structured Data](#augment-structured-data) flow.

### Input Format
This flow expects the INPUT_FILE key to be the key for the input dictionary. The value for this key should be the path to the raw text file (e.g. `.txt` or `.html`) from which you want to generate the QA pairs.

```
input_dict = {INPUT_FILE: your_input_file_path}
```

### Process
Then you can run the flow on the input dictionaries:
```
input_list = [input_dict]
output_list = client.run(input_list)
```

First, the file is run through the [Generate Structured Data from Unstructured Text process](#process-1). Then, the output QA pairs are run through the [Augment Structured Data process](#process).

### Output
The output Question-Answer pairs generated from the raw text are saved to `output_qa_text_data.csv` file in the `./data/output` folder.

The augmented output Question-Answer pairs are saved to `csv` and `json` files called `output_qa_augment_data` in the `./data/output` folder.

