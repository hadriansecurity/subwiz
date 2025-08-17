"""DNS resolution module for checking domain registration status.

This module provides asynchronous DNS resolution functionality to check whether
domains are registered and resolve to IP addresses. It uses multiple nameservers
for reliability and implements concurrency control for efficient batch processing.
"""

import asyncio

import aiodns
import idna.core

from subwiz.type import Domain


NAME_SERVERS = ["1.1.1.1", "1.0.0.1", "8.8.8.8"]
TIMEOUT = 3
TRIES = 1


async def get_registered_domains(
    domains_to_check: set[Domain], resolution_concurrency: int
) -> set[Domain]:
    """Check which domains from a set are registered and resolve to IP addresses.

    Args:
        domains_to_check: Set of Domain objects to check for registration
        resolution_concurrency: Maximum number of concurrent DNS resolutions

    Returns:
        Set of Domain objects that are registered and resolve successfully
    """

    semaphore = asyncio.Semaphore(resolution_concurrency)
    resolver = aiodns.DNSResolver(
        nameservers=NAME_SERVERS, timeout=TIMEOUT, tries=TRIES
    )

    domains_list = list(domains_to_check)
    tasks = [dom.is_registered(resolver, semaphore) for dom in domains_to_check]
    results = await asyncio.gather(*tasks)

    return {dom for dom, is_reg in zip(domains_list, results) if is_reg}
