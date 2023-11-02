# Generate Structured Data from Unstructured Text (Self Instructed)
Generate data from html text, and then fine tune the model on the generated data using the `Client("flow_self_instructed_gen")` interface. The [`demo_self_instruct_ft` notebook](demo_self_instruct_ft.ipynb) shows how to use `uniflow` to generate QA pairs from raw html with this flow.

### Setup
To get started, import Client and the input keys from `uniflow`:
```
from uniflow.client import Client
from uniflow.flow.constants import (OUTPUT_NAME, INPUT_FILE)
```

Then, initiate a `Client` flow for `flow_self_instructed_gen`:
```
client = Client("flow_self_instructed_gen")
```

### Input Format
This flow expects the INPUT_FILE key to be the key for the input dictionary. The value for this key should be the path to the `.html` file from which you want to generate the QA pairs.

```
input_dict = {INPUT_FILE: your_input_file_path}
```

### Process
Then you can run the flow on the input dictionaries:
```
input_list = [input_dict]
output_list = client.run(input_list)
```
The raw html is processed and split. Then, `uniflow` untilizes the [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) HuggingFace model to generate one question per paragraph of content.

### Output
The output Question-Answer pairs are saved to `output_self_instructed_data.csv` file in the `./data/output` folder.