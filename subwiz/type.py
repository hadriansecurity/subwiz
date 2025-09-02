"""Type definitions and validation functions for the subwiz package.

This module provides the Domain class for representing and validating domain names,
along with various type validation functions for command-line arguments and
configuration parameters.
"""

import argparse
import asyncio
import os
from typing import Optional, Union

import aiodns
import idna.core
import tldextract
import torch


class Domain:
    """Represents a domain name with validation and DNS resolution capabilities."""

    def __init__(self, value: str):
        """Initialize a Domain object from a string representation.

        Args:
            value: String representation of the domain

        Raises:
            ValueError: If the value is not a valid domain format
        """
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
        """Get the subdomain part of the domain.

        Returns:
            Subdomain string (empty string if no subdomain)
        """
        return self._subdomain

    @property
    def apex_domain(self) -> str:
        """Get the apex domain (domain + suffix without subdomain).

        Returns:
            Apex domain string
        """
        return self._domain + "." + self._suffix

    def __str__(self):
        """String representation of the domain.

        Returns:
            Full domain string including subdomain if present
        """
        if not self._subdomain:
            return self.apex_domain
        return self._subdomain + "." + self.apex_domain

    def __hash__(self):
        """Hash value for the domain object.

        Returns:
            Hash value based on string representation
        """
        return hash(str(self))

    def __eq__(self, other):
        """Equality comparison with another domain object.

        Args:
            other: Object to compare with

        Returns:
            True if domains are equal, False otherwise
        """
        return str(self) == str(other)

    async def is_registered(
        self, resolver: aiodns.DNSResolver, semaphore: asyncio.Semaphore
    ) -> bool:
        """Check if the domain is registered and resolves to an IP address.

        Args:
            resolver: DNS resolver instance to use for queries
            semaphore: Semaphore for controlling concurrency

        Returns:
            True if domain resolves successfully, False otherwise
        """
        async with semaphore:
            try:
                await resolver.query(str(self), "A")
                return True
            except idna.IDNAError:
                return False
            except aiodns.error.DNSError:
                return False


def max_recursion_type(value: str | int) -> int:
    """Validate and convert max recursion value.

    Args:
        value: String or integer value to validate

    Returns:
        Validated integer value

    Raises:
        argparse.ArgumentTypeError: If value is invalid or out of range
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use an integer >= 0 and <= 50: {value}")
    # don't allow > 50, so that model does not generate forever for wild card
    if not 0 <= ivalue <= 50:
        raise argparse.ArgumentTypeError(f"use an integer >= 0 and <= 50 {value}")
    return ivalue


def positive_int_type(value: str | int) -> int:
    """Validate and convert positive integer value.

    Args:
        value: String or integer value to validate

    Returns:
        Validated positive integer value

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use a positive integer: {value}")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"use a positive integer: {value}")
    return ivalue


def input_domains_file_type(value: str | os.PathLike) -> list[Domain]:
    """Validate and load domains from a file.

    Args:
        value: Path to file containing domain names

    Returns:
        List of Domain objects loaded from file

    Raises:
        argparse.ArgumentTypeError: If file doesn't exist or contains invalid domains
    """
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
    """Validate and convert list of domain strings to Domain objects.

    Args:
        value: List of domain strings to validate

    Returns:
        List of validated Domain objects

    Raises:
        argparse.ArgumentTypeError: If list is empty, contains invalid domains, or no subdomains
    """
    if len(value) == 0:
        raise argparse.ArgumentTypeError("empty input domains")

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

    return domains


def output_file_type(value: Optional[Union[str, os.PathLike]]) -> Optional[os.PathLike]:
    """Validate output file path.

    Args:
        value: File path to validate, or None

    Returns:
        Validated file path or None

    Raises:
        argparse.ArgumentTypeError: If path is not pathlike
    """
    if value is None:
        return value

    if not isinstance(value, (os.PathLike, str)):
        raise argparse.ArgumentTypeError(
            f"use a pathlike input for file path:  {value}"
        )
    return value


def temperature_type(value: str | float) -> float:
    """Validate and convert temperature value.

    Args:
        value: String or float value to validate

    Returns:
        Validated float value between 0 and 1

    Raises:
        argparse.ArgumentTypeError: If value is not a valid float in range [0, 1]
    """
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid float: {value}")
    if not 0 <= fvalue <= 1:
        raise argparse.ArgumentTypeError(f"use 0 ≤ temperature ≤ 1: {value}")
    return fvalue


def concurrency_type(value: str | int) -> int:
    """Validate and convert concurrency value.

    Args:
        value: String or integer value to validate

    Returns:
        Validated integer value between 1 and 256

    Raises:
        argparse.ArgumentTypeError: If value is not a valid integer in range [1, 256]
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid int: {value}")
    if not 1 <= ivalue <= 256:
        raise argparse.ArgumentTypeError(f"use 1 ≤ concurrency ≤ 256: {value}")
    return ivalue


def device_type(value: str) -> str:
    """Validate and determine device type for model execution.

    Args:
        value: Device string to validate

    Returns:
        Validated device string or auto-detected device

    Raises:
        argparse.ArgumentTypeError: If value is not a valid device string
    """
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("use a string for the device")

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
