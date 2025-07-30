import argparse
import asyncio
import os
import re
from collections import defaultdict
from typing import Callable

from huggingface_hub import hf_hub_download
import torch
from transformers import PreTrainedTokenizerFast

from subwiz.cli_printer import print_hello, print_log, print_progress_dot
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
    """Download files from HuggingFace to run subwiz. Caches in local file system."""

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
    """Load model, preprocess inputs, tokenize text, run inference and decode tokens back to text."""

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
    """Check whether predictions resolve."""

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
    multi_apex: bool = False,
    print_cli_progress: bool = False,
) -> list[str]:
    """Process inputs, download model, run inference, check if predictions resolve and return hits."""
    if print_cli_progress:
        print_hello()

    domain_objects = input_domains_type(input_domains)
    device = device_type(device)
    num_predictions = positive_int_type(num_predictions)
    max_new_tokens = positive_int_type(max_new_tokens)
    temperature = temperature_type(temperature)
    resolution_concurrency = concurrency_type(resolution_concurrency)

    domain_groups = defaultdict(list)
    for dom in domain_objects:
        domain_groups[dom.apex_domain].append(dom)

    if len(domain_groups) > 1 and not multi_apex:
        raise argparse.ArgumentTypeError(
            f"multiple apex domains found: {sorted(domain_groups.keys())}. "
            "Use the --multi-apex flag to process them all."
        )

    model_path, tokenizer_path = download_files(force_download)

    all_predictions = set()

    for apex, domains_in_group in domain_groups.items():

        on_inference_iteration = None
        if print_cli_progress:
            on_inference_iteration = print_progress_dot
            log = f"running inference for {apex}" if multi_apex else "running inference"
            print_log(log, end="")

        predictions = run_inference(
            input_domains=domains_in_group,
            device=device,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            num_predictions=num_predictions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            on_inference_iteration=on_inference_iteration,
        )
        print_log("", end="\n")
        all_predictions.update(predictions)

    if no_resolve:
        return sorted(list(all_predictions))

    resolving_predictions = run_resolution(
        sorted(list(all_predictions)), resolution_lim=resolution_concurrency
    )
    return resolving_predictions
