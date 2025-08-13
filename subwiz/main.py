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
from subwiz.resolve import get_registered_domains
from subwiz.type import (
    Domain,
    input_domains_type,
    device_type,
    max_recursion_type,
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
    blocked_domains: set[Domain],
    on_inference_iteration: Callable = None,
) -> set[Domain]:
    """Preprocess inputs, tokenize text, run inference and decode tokens back to text."""

    apex = next(iter(input_domains)).apex_domain
    subs = [dom.subdomain for dom in input_domains]
    tokenizer_input = ",".join(sorted(subs)) + "[DELIM]"
    # TODO: pick a different subset, if some were out of context last iteration

    x = tokenizer.encode(tokenizer_input)
    x = [1] * (gpt_model.config.block_size - len(x)) + x
    x = torch.tensor(x)

    blocked_outputs = {dom.subdomain for dom in blocked_domains}

    predictions = gpt_model.generate(
        x,
        max_new_tokens=max_new_tokens,
        topn=num_predictions,
        on_iteration=on_inference_iteration,
        temperature=temperature,
        blocked_outputs=blocked_outputs,
    )
    predictions = predictions.int().tolist()

    predictions = {
        tokenizer.decode(pred).replace(" ", "").rsplit("[DELIM]", 1)[1]
        for pred in predictions
    }

    predictions: set[str] = {sub + "." + apex for sub in predictions}

    predicted_domains: set[Domain] = set()
    for pred in predictions:
        try:
            predicted_domains.add(Domain(pred))
        except ValueError:  # invalid domain name
            pass

    return predicted_domains


def _get_domains_for_group(
    domains_in_group: set[Domain],
    all_apexes: set[str],
    gpt_model: GPT,
    tokenizer: PreTrainedTokenizerFast,
    multi_apex: bool,
    num_predictions: int,
    max_new_tokens: int,
    max_recursion: int,
    temperature: float,
    no_resolve: bool,
    resolution_concurrency: int,
    print_cli_progress: bool,
) -> set[str]:
    """For a group of subdomains that share an apex: run inference and check if they resolve, recursively."""

    blocked_domains: set[Domain] = domains_in_group.copy()
    all_predictions_that_resolve: set[Domain] = set()
    apex = next(iter(domains_in_group)).apex_domain

    for i in range(max_recursion):

        on_inference_iteration = None
        if print_cli_progress:

            dots_emitted = 0

            def _counting_progress_dot():
                nonlocal dots_emitted
                dots_emitted += 1
                print_progress_dot()

            on_inference_iteration = _counting_progress_dot

            log = "running inference"
            if multi_apex:
                log += f" for {apex}"
            if i:
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
            blocked_domains=blocked_domains,
        )

        if no_resolve:
            print_log("", end="\n")
            return {str(dom) for dom in predictions}

        predictions_that_resolve = asyncio.run(
            get_registered_domains(predictions, resolution_concurrency)
        )

        if print_cli_progress:
            if not max_recursion:
                end_log = ""
            else:
                printed_characters = len(apex) + dots_emitted + (0 if i == 0 else 3)
                max_characters = max(len(a) for a in all_apexes) + max_new_tokens + 3
                extra_spaces = max_characters - printed_characters + 1

                sub_count = len(predictions_that_resolve)
                subs_label = "subdomain" if sub_count == 1 else "subdomains"
                end_log = " " * extra_spaces + f"found {sub_count:>3} {subs_label:>10}"
            print_log(end_log, end="\n")

        if not predictions_that_resolve:
            break

        all_predictions_that_resolve |= predictions_that_resolve

        blocked_domains |= predictions
        domains_in_group |= predictions_that_resolve

    return {str(dom) for dom in all_predictions_that_resolve}


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
    max_recursion: int = 5,
    print_cli_progress: bool = False,
) -> list[str]:
    """Check types, download model, get new subdomains for each apex."""
    if print_cli_progress:
        print_hello()

    domain_objects = input_domains_type(input_domains)
    device = device_type(device)
    num_predictions = positive_int_type(num_predictions)
    max_new_tokens = positive_int_type(max_new_tokens)
    max_recursion = max_recursion_type(max_recursion)
    temperature = temperature_type(temperature)
    resolution_concurrency = concurrency_type(resolution_concurrency)

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

    for apex in sorted(domain_groups):
        found_domains |= _get_domains_for_group(
            domains_in_group=domain_groups[apex],
            all_apexes=set(domain_groups.keys()),
            gpt_model=gpt_model,
            tokenizer=tokenizer,
            multi_apex=multi_apex,
            num_predictions=num_predictions,
            max_new_tokens=max_new_tokens,
            max_recursion=max_recursion,
            temperature=temperature,
            no_resolve=no_resolve,
            resolution_concurrency=resolution_concurrency,
            print_cli_progress=print_cli_progress,
        )

    return sorted(found_domains)
