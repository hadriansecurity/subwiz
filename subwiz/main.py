"""Main module for running subdomain enumeration using transformer models.

This module provides the core functionality for downloading models, running inference,
and orchestrating the subdomain discovery process. It handles model loading,
tokenization, inference execution, and result processing.
"""

import argparse
import asyncio
import os
import re
from collections import defaultdict
from typing import Callable

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
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
MODEL_FILE = "model_v2.pt"
TOKENIZER_FILE = "tokenizer_v2.json"
CONFIG_FILE = "config.json"


def get_model_and_tokenizer(
    force_download: bool,
    device: str,
    quiet: bool,
) -> tuple[GPT, PreTrainedTokenizerFast]:
    """Download files from HuggingFace to run subwiz. Caches in local file system.

    Args:
        force_download: Whether to force download even if files exist locally
        device: Device to load the model on

    Returns:
        Tuple of (GPT model, tokenizer)
    """
    if quiet:
        disable_progress_bars()

    model_path = hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE, force_download=force_download
    )
    tokenizer_path = hf_hub_download(
        repo_id=MODEL_REPO, filename=TOKENIZER_FILE, force_download=force_download
    )
    hf_hub_download(
        repo_id=MODEL_REPO, filename=CONFIG_FILE, force_download=force_download
    )
    if quiet:
        enable_progress_bars()

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
    """Preprocess inputs, tokenize text, run inference and decode tokens back to text.

    Args:
        input_domains: Set of input domain objects
        gpt_model: Loaded GPT model for inference
        tokenizer: Tokenizer for text processing
        num_predictions: Number of predictions to generate
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature for generation
        blocked_domains: Domains that should not be generated
        on_inference_iteration: Optional callback for progress tracking

    Returns:
        Set of predicted domain objects
    """

    subs = [dom.subdomain for dom in input_domains]
    apex_domain = input_domains[0].apex_domain
    subdomains_tokenizer_input = ",".join(sorted(subs)) + "[DELIM]"
    apex_tokenizer_input = "[BOS]" + apex_domain + "[DELIM]"

    subs_x = tokenizer.encode(subdomains_tokenizer_input)
    apex_x = tokenizer.encode(apex_tokenizer_input)

    # Trim subs to account for the apex part, grab last part
    subs_x = subs_x[:gpt_model.config.block_size - len(apex_x)]

    x = apex_x + subs_x
    x = [gpt_model.pad_token] * (gpt_model.config.block_size - len(x)) + x
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

    predictions: set[str] = {sub + "." + apex_domain for sub in predictions}

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
    quiet: bool,
) -> set[str]:
    """For a group of subdomains that share an apex: run inference and check if they resolve, recursively.

    Args:
        domains_in_group: Set of domains sharing the same apex
        all_apexes: Set of all apex domains for progress display
        gpt_model: Loaded GPT model for inference
        tokenizer: Tokenizer for text processing
        multi_apex: Whether multiple apex domains are being processed
        num_predictions: Number of predictions to generate
        max_new_tokens: Maximum new tokens to generate
        max_recursion: Maximum recursion depth for discovery
        temperature: Sampling temperature for generation
        no_resolve: Whether to skip DNS resolution
        resolution_concurrency: Number of concurrent DNS resolutions
        print_cli_progress: Whether to print progress information

    Returns:
        Set of discovered subdomain strings
    """

    blocked_domains: set[Domain] = domains_in_group.copy()
    all_predictions_that_resolve: set[Domain] = set()
    apex = next(iter(domains_in_group)).apex_domain

    for i in range(max_recursion):

        on_inference_iteration = None
        if not quiet:

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
            if not quiet:
                print_log("", end="\n")
            return {str(dom) for dom in predictions}

        predictions_that_resolve = asyncio.run(
            get_registered_domains(predictions, resolution_concurrency)
        )

        if not quiet:
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
    quiet: bool = True,
) -> list[str]:
    """Check types, download model, get new subdomains for each apex.

    Args:
        input_domains: List of input domain strings
        device: Hardware device to run the model on
        num_predictions: Number of subdomains to predict
        max_new_tokens: Maximum length of predicted subdomains in tokens
        temperature: Sampling temperature for generation
        resolution_concurrency: Number of concurrent DNS resolutions
        no_resolve: Whether to skip DNS resolution
        force_download: Whether to force download model files
        multi_apex: Whether to allow multiple apex domains
        max_recursion: Maximum recursion depth for discovery
        print_cli_progress: Whether to print progress information

    Returns:
        List of discovered subdomain strings

    Raises:
        argparse.ArgumentTypeError: If multiple apex domains found without multi_apex flag
    """
    if not quiet:
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

    gpt_model, tokenizer = get_model_and_tokenizer(
        force_download, device=device, quiet=quiet
    )
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
            quiet=quiet,
        )

    return sorted(found_domains)
