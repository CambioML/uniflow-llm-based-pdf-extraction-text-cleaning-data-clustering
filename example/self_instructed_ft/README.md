# Generate Structured Data from Unstructured Text (Self Instructed)
Generate data from html text, and then fine tune the model on the generated data using the `Client("flow_self_instructed_gen")` interface. The [`demo_self_instruct_ft` notebook](demo_self_instruct_ft.ipynb) shows how to use `uniflow` to generate QA pairs from raw html with this flow.

### Input Format
This flow expects the HTML_FILE key to be the key for the input dictionary. The value for this key should be the path to the `.html` file from which you want to generate the QA pairs.

```
input_dict = {HTML_FILE: your_input_file_path}
```

### Process
The raw html is processed and split. Then, `uniflow` untilizes the [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) HuggingFace model to generate one question per paragraph of content.

### Output
The output Question-Answer pairs are saved to `QA_output_self-instruct.csv` file in the `./data/output` folder.