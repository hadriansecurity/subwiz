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


MODEL_REPO = "HadrianSecurity/subwiz"
MODEL_FILE = "model.pt"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"


class Domain:
    def __init__(self, value: str):
        extract = tldextract.extract(value)
        self.subdomain, self.domain, self.suffix = (
            extract.subdomain,
            extract.domain,
            extract.suffix,
        )

        if not bool(self.domain and self.suffix):
            raise ValueError(f"not a valid domain: {value}")

    @property
    def apex_domain(self) -> str:
        return self.domain + "." + self.suffix

    def __str__(self):
        if not self.subdomain:
            return self.apex_domain
        return self.subdomain + "." + self.apex_domain


def positive_int_validator(value: str | int) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use a positive integer: {value}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"use a positive integer: {value}")
    return ivalue


def input_domains_file_validator(value: str | os.PathLike) -> list[Domain]:
    if not isinstance(value, (os.PathLike, str)):
        raise argparse.ArgumentTypeError(
            f"use a pathlike input for file path:  {value}"
        )
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f"file not found: {value}")

    with open(value) as f:
        input_domains = f.read().split()

    return input_domains_validator(input_domains)


def input_domains_validator(value: list[str]) -> list[Domain]:
    if len(value) == 0:
        raise argparse.ArgumentTypeError(f"empty input domains")

    value = set(value)

    domains, invalid_domains = [], []
    for dom in value:
        try:
            domains.append(Domain(dom))
        except ValueError:
            invalid_domains.append(dom)

    if invalid_domains:
        raise argparse.ArgumentTypeError(
            f"invalid input domains: {sorted(invalid_domains)}"
        )

    apex_domains = sorted({dom.apex_domain for dom in domains})
    if len(apex_domains) != 1:
        raise argparse.ArgumentTypeError(
            f"all input domains must have same apex. found: {apex_domains}"
        )

    if not any(dom.subdomain for dom in domains):
        raise argparse.ArgumentTypeError(
            f"input should include at least one subdomain: {list(value)}"
        )

    return domains


def temperature_validator(value: str | float) -> float:
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid float: {value}")
    if not 0 <= fvalue <= 1:
        raise argparse.ArgumentTypeError(f"use 0 ≤ temperature ≤ 1: {value}")
    return fvalue


def concurrency_validator(value: str | int) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid int: {value}")
    if not 1 <= ivalue <= 256:
        raise argparse.ArgumentTypeError(f"use 1 ≤ concurrency ≤ 256: {value}")
    return value


def device_validator(value: str) -> str:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError(f"use a string for the device")

    if value not in ["auto", "cpu", "cuda", "mps"]:
        raise argparse.ArgumentTypeError(
            f'device should be in ["auto", "cpu", "cuda", "mps"]: {value}'
        )

    if value != "auto":
        return value

    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, clean_up_tokenization_spaces=True)

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
):
    input_domains = input_domains_validator(input_domains)
    device = device_validator(device)
    num_predictions = positive_int_validator(num_predictions)
    max_new_tokens = positive_int_validator(max_new_tokens)
    temperature = temperature_validator(temperature)
    resolution_concurrency = concurrency_validator(resolution_concurrency)

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
