"""DNS resolution module for checking domain registration status.

This module provides asynchronous DNS resolution functionality to check whether
domains are registered and resolve to IP addresses. It uses multiple nameservers
for reliability and implements concurrency control for efficient batch processing.

It also provides wildcard-DNS detection. Domains with a wildcard record
(``*.example.com``) resolve *every* possible subdomain to the same catch-all
IP set, so plain "does it resolve?" checks produce huge numbers of false
positives. ``detect_wildcard`` finds the catch-all IP set up front so that
resolution can filter to only the subdomains that resolve *outside* it.
"""

from __future__ import annotations

import asyncio
import random
import string

import aiodns
import idna.core

from subwiz.type import Domain

NAME_SERVERS = ["1.1.1.1", "1.0.0.1", "8.8.8.8"]
TIMEOUT = 3
TRIES = 1

WILDCARD_PROBE_COUNT = 3
WILDCARD_LABEL_LENGTH = 20


def _new_resolver() -> aiodns.DNSResolver:
    """Create a DNS resolver configured with the module nameservers."""
    return aiodns.DNSResolver(nameservers=NAME_SERVERS, timeout=TIMEOUT, tries=TRIES)


async def _resolve_ips(
    hostname: str, resolver: aiodns.DNSResolver, semaphore: asyncio.Semaphore
) -> set[str]:
    """Resolve a hostname to its set of IPv4 addresses.

    Args:
        hostname: Hostname to resolve
        resolver: DNS resolver instance to use for queries
        semaphore: Semaphore for controlling concurrency

    Returns:
        Set of IP address strings the hostname resolves to, or an empty set if
        it does not resolve.
    """
    async with semaphore:
        try:
            results = await resolver.query(hostname, "A")
            return {record.host for record in results}
        except idna.IDNAError:
            return set()
        except aiodns.error.DNSError:
            return set()


def _random_label() -> str:
    """Generate a random subdomain label unlikely to exist as a real record."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=WILDCARD_LABEL_LENGTH))


async def detect_wildcard(
    apex_domain: str, resolution_concurrency: int = WILDCARD_PROBE_COUNT
) -> set[str]:
    """Detect a wildcard DNS record on an apex domain.

    Probes several random subdomains that should not exist. If they all
    resolve, the apex has a wildcard (catch-all) record and the union of the
    IPs they resolve to is returned as the wildcard IP set.

    Args:
        apex_domain: Apex domain to probe (e.g. ``example.com``)
        resolution_concurrency: Maximum number of concurrent DNS resolutions

    Returns:
        The set of catch-all IPs if a wildcard is present, otherwise an empty
        set.
    """
    semaphore = asyncio.Semaphore(resolution_concurrency)
    resolver = _new_resolver()

    probes = [f"{_random_label()}.{apex_domain}" for _ in range(WILDCARD_PROBE_COUNT)]
    probe_ips = await asyncio.gather(
        *[_resolve_ips(probe, resolver, semaphore) for probe in probes]
    )

    if not all(probe_ips):
        return set()

    return set().union(*probe_ips)


async def get_registered_domains(
    domains_to_check: set[Domain],
    resolution_concurrency: int,
    wildcard_ips: set[str] | None = None,
) -> set[Domain]:
    """Check which domains from a set are registered and resolve to IP addresses.

    When ``wildcard_ips`` is provided (the apex has a wildcard record), a
    domain counts as registered only if it resolves to at least one IP
    *outside* the wildcard set. Domains resolving solely to the catch-all IPs
    are treated as false positives and dropped.

    Args:
        domains_to_check: Set of Domain objects to check for registration
        resolution_concurrency: Maximum number of concurrent DNS resolutions
        wildcard_ips: Catch-all IPs of a detected wildcard, or None/empty for
            the standard "resolves = registered" check.

    Returns:
        Set of Domain objects that are registered and resolve successfully
    """

    semaphore = asyncio.Semaphore(resolution_concurrency)
    resolver = _new_resolver()

    domains_list = list(domains_to_check)

    if not wildcard_ips:
        tasks = [dom.is_registered(resolver, semaphore) for dom in domains_list]
        results = await asyncio.gather(*tasks)
        return {dom for dom, is_reg in zip(domains_list, results) if is_reg}

    tasks = [_resolve_ips(str(dom), resolver, semaphore) for dom in domains_list]
    results = await asyncio.gather(*tasks)
    return {dom for dom, ips in zip(domains_list, results) if ips - wildcard_ips}
