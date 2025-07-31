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


def get_model_and_tokenizer(
    force_download: bool,
    device: str,
) -> tuple[GPT, PreTrainedTokenizerFast]:
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

    gpt_model = GPT.from_checkpoint(
        model_path, device=device, tokenizer_path=tokenizer_path
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, clean_up_tokenization_spaces=True
    )

    return gpt_model, tokenizer


def run_inference(
    input_domains: set[Domain],
    gpt_model: GPT,
    tokenizer: PreTrainedTokenizerFast,
    num_predictions: int,
    max_new_tokens: int,
    temperature: float,
    on_inference_iteration: Callable = None,
) -> set[str]:
    """Preprocess inputs, tokenize text, run inference and decode tokens back to text."""

    subs = [dom.subdomain for dom in input_domains]
    subs.sort()
    tokenizer_input = ",".join(subs) + "[DELIM]"

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
    predictions = {
        sub + "." + next(iter(input_domains)).apex_domain for sub in predictions
    }
    predictions = {re.sub(r"\.+", ".", dom) for dom in predictions}
    predictions = predictions - {str(dom) for dom in input_domains}

    return predictions


def run_resolution(predictions: set[str], resolution_lim: int) -> set[str]:
    """Check whether predictions resolve."""

    registered_domains = asyncio.run(is_registered_bulk(predictions, resolution_lim))
    return registered_domains


def _get_domains_for_group(
    domains_in_group: set[Domain],
    gpt_model: GPT,
    tokenizer: PreTrainedTokenizerFast,
    multi_apex: bool,
    num_predictions: int,
    max_new_tokens: int,
    temperature: float,
    no_recursion: bool,
    no_resolve: bool,
    resolution_lim: int,
    print_cli_progress: bool,
) -> set[str]:
    """For a group of subdomains that share an apex: run inference and check if they resolve, recursively."""

    all_predictions_that_resolve = set()
    apex = next(iter(domains_in_group)).apex_domain

    max_recursion_iters = 1 if no_recursion else 20
    for i in range(max_recursion_iters):

        on_inference_iteration = None
        if print_cli_progress:
            on_inference_iteration = print_progress_dot
            log = "running inference"
            if multi_apex:
                log += f" for {apex}"
            if not no_recursion and i:
                log += f" x{i + 1}"
            print_log(log, end="")

        predictions = run_inference(
            input_domains=domains_in_group,
            gpt_model=gpt_model,
            tokenizer=tokenizer,
            num_predictions=num_predictions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            on_inference_iteration=on_inference_iteration,
        )

        if no_resolve:
            print_log("", end="\n")
            return predictions

        predictions_that_resolve = run_resolution(predictions, resolution_lim)
        end_log = "" if no_recursion else "done"
        print_log(end_log, end="\n")

        if not predictions_that_resolve:
            break

        all_predictions_that_resolve |= predictions_that_resolve
        domains_in_group |= {Domain(dom) for dom in predictions_that_resolve}

    return all_predictions_that_resolve


def run(
    input_domains: list[str],
    device: str = "auto",
    num_predictions: int = 500,
    max_new_tokens: int = 10,
    temperature: float = 0,
    resolution_lim: int = 128,
    no_resolve: bool = False,
    force_download: bool = False,
    multi_apex: bool = False,
    no_recursion: bool = True,
    print_cli_progress: bool = False,
) -> list[str]:
    """Check types, download model, get new subdomains for each apex."""
    if print_cli_progress:
        print_hello()

    domain_objects = input_domains_type(input_domains)
    device = device_type(device)
    num_predictions = positive_int_type(num_predictions)
    max_new_tokens = positive_int_type(max_new_tokens)
    temperature = temperature_type(temperature)
    resolution_lim = concurrency_type(resolution_lim)

    domain_groups = defaultdict(set)
    for dom in domain_objects:
        domain_groups[dom.apex_domain].add(dom)

    if len(domain_groups) > 1 and not multi_apex:
        raise argparse.ArgumentTypeError(
            f"multiple apex domains found: {sorted(domain_groups.keys())}. "
            "Use the --multi-apex flag to process them all."
        )

    gpt_model, tokenizer = get_model_and_tokenizer(force_download, device=device)
    found_domains = set()

    for apex, domains_in_group in domain_groups.items():
        found_domains |= _get_domains_for_group(
            domains_in_group=domains_in_group,
            gpt_model=gpt_model,
            tokenizer=tokenizer,
            multi_apex=multi_apex,
            num_predictions=num_predictions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            no_recursion=no_recursion,
            no_resolve=no_resolve,
            resolution_lim=resolution_lim,
            print_cli_progress=print_cli_progress,
        )

    return sorted(found_domains)
