import argparse
import asyncio
import os
import re
from typing import Callable

from huggingface_hub import hf_hub_download
import tldextract
import torch
from transformers import PreTrainedTokenizerFast

from subwiz.model import GPT
from subwiz.resolve import is_registered_bulk
from subwiz.type import (
    Domain,
    input_domains_type,
    device_type,
    positive_int_type,
    temperature_type,
    concurrency_type,
)


MODEL_REPO = "HadrianSecurity/subwiz"
MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"


def download_files(force_download: bool) -> tuple[str, str]:
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE, force_download=force_download
    )
    tokenizer_path = hf_hub_download(
        repo_id=MODEL_REPO, filename=TOKENIZER_FILE, force_download=force_download
    )
    hf_hub_download(
        repo_id=MODEL_REPO, filename=CONFIG_FILE, force_download=force_download
    )
    return model_path, tokenizer_path


def run_inference(
    input_domains: list[Domain],
    device: str,
    model_path: str,
    tokenizer_path: str,
    num_predictions: int,
    max_new_tokens: int,
    temperature: float,
    on_inference_iteration: Callable = None,
) -> list[str]:

    gpt_model = GPT.from_checkpoint(
        model_path, device=device, tokenizer_path=tokenizer_path
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, clean_up_tokenization_spaces=True
    )

    subs = [dom.subdomain for dom in input_domains]
    tokenizer_input = ",".join(sorted(subs)) + "[DELIM]"

    x = tokenizer.encode(tokenizer_input)
    x = [1] * (gpt_model.config.block_size - len(x)) + x
    x = torch.tensor(x)

    predictions = gpt_model.generate(
        x,
        max_new_tokens=max_new_tokens,
        topn=num_predictions,
        on_iteration=on_inference_iteration,
        temperature=temperature,
    )
    predictions = predictions.int().tolist()

    predictions = {
        tokenizer.decode(pred).replace(" ", "").rsplit("[DELIM]", 1)[1]
        for pred in predictions
    }
    predictions = {sub + "." + input_domains[0].apex_domain for sub in predictions}
    predictions = {re.sub(r"\.+", ".", dom) for dom in predictions}
    predictions = predictions - {str(dom) for dom in input_domains}

    return sorted(predictions)


def run_resolution(predictions: list[str], resolution_lim: int):
    registered_domains = asyncio.run(is_registered_bulk(predictions, resolution_lim))
    return registered_domains


def run(
    input_domains: list[str],
    device: str = "auto",
    num_predictions: int = 500,
    max_new_tokens: int = 10,
    temperature: float = 0,
    resolution_concurrency: int = 128,
    no_resolve: bool = False,
    force_download: bool = False,
) -> list[str]:

    input_domains = input_domains_type(input_domains)
    device = device_type(device)
    num_predictions = positive_int_type(num_predictions)
    max_new_tokens = positive_int_type(max_new_tokens)
    temperature = temperature_type(temperature)
    resolution_concurrency = concurrency_type(resolution_concurrency)

    model_path, tokenizer_path = download_files(force_download)

    predictions = run_inference(
        input_domains=input_domains,
        device=device,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        num_predictions=num_predictions,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if no_resolve:
        return predictions

    resolving_predictions = run_resolution(
        predictions, resolution_lim=resolution_concurrency
    )
    return resolving_predictions
