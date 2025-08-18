import argparse
import asyncio
import os
from typing import Optional, Union

import aiodns
import idna.core
import tldextract
import torch


class Domain:
    def __init__(self, value: str):
        extract = tldextract.extract(value.lower())

        if not (extract.domain and extract.suffix):
            raise ValueError(f"not a valid domain: {value}")

        self._subdomain, self._domain, self._suffix = (
            extract.subdomain,
            extract.domain,
            extract.suffix,
        )

    @property
    def subdomain(self) -> str:
        return self._subdomain

    @property
    def apex_domain(self) -> str:
        return self._domain + "." + self._suffix

    def __str__(self):
        return (self.subdomain + "." if self.subdomain else "") + self.apex_domain

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    async def is_registered(
        self, resolver: aiodns.DNSResolver, semaphore: asyncio.Semaphore
    ) -> bool:
        async with semaphore:
            try:
                await resolver.query(str(self), "A")
                return True
            except idna.IDNAError:
                return False
            except aiodns.error.DNSError:
                return False


def max_recursion_type(value: str | int) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use an integer >= 0 and <= 50: {value}")
    # don't allow > 50, so that model does not generate forever for wild card
    if not 0 <= ivalue <= 50:
        raise argparse.ArgumentTypeError(f"use an integer >= 0 and <= 50 {value}")
    return ivalue


def positive_int_type(value: str | int) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use a positive integer: {value}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"use a positive integer: {value}")
    return ivalue


def input_domains_file_type(value: str | os.PathLike) -> list[Domain]:
    if not isinstance(value, (os.PathLike, str)):
        raise argparse.ArgumentTypeError(
            f"use a pathlike input for file path:  {value}"
        )
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f"file not found: {value}")

    with open(value) as f:
        input_domains = f.read().split()

    return input_domains_type(input_domains)


def input_domains_type(value: list[str]) -> list[Domain]:
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

    if not any(dom.subdomain for dom in domains):
        raise argparse.ArgumentTypeError(
            f"input should include at least one subdomain: {list(value)}"
        )

    return domains


def output_file_type(value: Optional[Union[str, os.PathLike]]) -> Optional[os.PathLike]:
    if value is None:
        return value

    if not isinstance(value, (os.PathLike, str)):
        raise argparse.ArgumentTypeError(
            f"use a pathlike input for file path:  {value}"
        )
    return value


def temperature_type(value: str | float) -> float:
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid float: {value}")
    if not 0 <= fvalue <= 1:
        raise argparse.ArgumentTypeError(f"use 0 ≤ temperature ≤ 1: {value}")
    return fvalue


def concurrency_type(value: str | int) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid int: {value}")
    if not 1 <= ivalue <= 256:
        raise argparse.ArgumentTypeError(f"use 1 ≤ concurrency ≤ 256: {value}")
    return ivalue


def device_type(value: str) -> str:
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
