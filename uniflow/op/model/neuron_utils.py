import hashlib
import os
import zipfile

import requests
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.mistral.model import MistralForSampling

neuron_model_config = {
    "mistralai/Mistral-7B-Instruct-v0.2": {
        1: {
            8192: [
                "https://cambioml-neuron-cache.s3.us-west-2.amazonaws.com/transformers_neuronx/Mistral-7B-Instruct-v0.2_tp2-bz1-8192.zip",
                "367aa7db73c1f82b5add0ad5c97d7a77",
            ]
        },
        2: {
            4096: [
                "https://cambioml-neuron-cache.s3.us-west-2.amazonaws.com/transformers_neuronx/Mistral-7B-Instruct-v0.2_tp2-bz2-4096.zip",
                "f2f07a6d6b05e39b7e9d94cd6c95fb96",
            ]
        },
        4: {
            2048: [
                "https://cambioml-neuron-cache.s3.us-west-2.amazonaws.com/transformers_neuronx/Mistral-7B-Instruct-v0.2_tp2-bz4-2048.zip",
                "1c62a06d6f25c2241b72b95e00ededfb",
            ]
        },
    }
}
instance_type_dict = {
    "inf2.xlarge": 2,
    "inf2.8xlarge": 2,
    "inf2.24xlarge": 12,
    "inf2.48xlarge": 24,
}


def get_instance_type():
    """
    Get the instance type of the current instance.
    """
    url = "http://169.254.169.254/latest/meta-data/instance-type"
    try:
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            return response.text
        else:
            return "Error: Unable to access metadata service"
    except Exception as e:
        return f"Error: {e}"


def verify_md5(file_path, expected_md5):
    """
    Verify the MD5 of a file against an expected MD5 hash.

    :param file_path: Path to the file to be checked
    :param expected_md5: Expected MD5 hash for the file
    :return: True if MD5 matches, False otherwise
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)
    file_md5 = md5_hash.hexdigest()
    return file_md5 == expected_md5


def download_and_unzip(url, expected_md5):
    """
    Download a file from a URL and unzip it.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/uniflow/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    zip_filename = os.path.join(cache_dir, f"{expected_md5}.zip")
    if os.path.exists(zip_filename):
        return os.path.join(cache_dir, f"{expected_md5}")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(zip_filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if not verify_md5(zip_filename, expected_md5) or expected_md5 != "":
        raise ValueError("MD5 verification failed")

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(os.path.join(cache_dir, f"{expected_md5}"))
    return os.path.join(cache_dir, f"{expected_md5}")


def split_neuron_model(model_name):
    """
    Split weights
    """
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
    )
    from transformers_neuronx.module import save_pretrained_split

    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache/uniflow/", model_name.split("/")[1] + "-split"
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    model_cpu = AutoModelForCausalLM.from_pretrained(model_name)
    save_pretrained_split(model_cpu, cache_dir)
    return cache_dir


def get_neuron_model(model_name, batch_size):
    """
    Get the neuron model for the given model name, batch size and n_positions.
    """
    instance_type = get_instance_type()
    assert "inf2" in instance_type, ValueError(
        "Neuron model can only run on the AWS EC2 inf2 instance series."
    )
    tp_degree = instance_type_dict[instance_type]
    assert model_name in neuron_model_config
    assert batch_size in neuron_model_config[model_name], ValueError(
        f"{model_name} only support batch size {list(neuron_model_config[model_name].keys())} in {instance_type}"
    )
    n_positions_list = list(neuron_model_config[model_name][batch_size].keys())
    n_positions_list.sort()
    n_positions = max(neuron_model_config[model_name][batch_size])
    neuron_config = NeuronConfig(grouped_query_attention=constants.GQA.SHARD_OVER_HEADS)

    model_split = split_neuron_model(model_name)

    model_neuron = MistralForSampling.from_pretrained(
        model_split,
        batch_size=batch_size,
        tp_degree=tp_degree,
        n_positions=n_positions,
        amp="bf16",
        neuron_config=neuron_config,
    )
    url, md5 = neuron_model_config[model_name][batch_size][n_positions]
    neuron_cache = download_and_unzip(url, md5)
    print("loading neuron model from cache, need to wait for a while...")
    model_neuron.load(neuron_cache)
    model_neuron.to_neuron()
    model_neuron.eval()
    model_config = AutoConfig.from_pretrained(model_name)
    model = HuggingFaceGenerationModelAdapter(model_config, model_neuron)
    model.reset_generation()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def batch_list(lst, batch_size):
    """
    Split a list into batches of a specified size.
    """
    batches = []
    for i in range(0, len(lst), batch_size):
        batch = lst[i : i + batch_size]
        while len(batch) < batch_size:
            batch.append(lst[-1])
        batches.append(batch)
    return batches


def neuron_infer(text_list, model, tokenizer):
    """
    Run neuron inference on a list of texts.
    """
    batches = batch_list(text_list, 4)
    results = []
    for batch in batches:
        batch = ["[INST]" + text + "[/INST]" for text in batch]
        encoded_input = tokenizer(batch, return_tensors="pt", padding=True)
        with torch.inference_mode():
            sample_output = model.generate(
                input_ids=encoded_input.input_ids,
                attention_mask=encoded_input.attention_mask,
                do_sample=True,
                max_length=1024,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
            )
        for sample_idx, tok in enumerate(sample_output):
            start = sum(encoded_input.input_ids[sample_idx] != tokenizer.eos_token_id)
            tok = tok[start + 1 :]
            end = list(tok).index(tokenizer.eos_token_id)
            results.append(tokenizer.decode(tok[:end]))
    results = results[: len(text_list)]
    return results
