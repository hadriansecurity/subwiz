"""DNS resolution module for checking domain registration status.

This module provides asynchronous DNS resolution functionality to check whether
domains are registered and resolve to IP addresses. It uses multiple nameservers
for reliability and implements concurrency control for efficient batch processing.
Includes wildcard DNS detection to filter false positives on domains with
catch-all DNS records.
"""

from __future__ import annotations

import asyncio
import random
import string
from typing import Optional

import aiodns
import idna.core

from subwiz.type import Domain


NAME_SERVERS = ["1.1.1.1", "1.0.0.1", "8.8.8.8"]
TIMEOUT = 3
TRIES = 1
WILDCARD_PROBE_COUNT = 3


async def _resolve_ips(
    hostname: str, resolver: aiodns.DNSResolver
) -> frozenset[str]:
    """Resolve a hostname to its set of IP addresses.

    Args:
        hostname: The hostname to resolve
        resolver: DNS resolver instance

    Returns:
        Frozenset of IP address strings, or empty frozenset on failure
    """
    try:
        results = await resolver.query(hostname, "A")
        return frozenset(r.host for r in results)
    except (aiodns.error.DNSError, idna.IDNAError):
        return frozenset()


async def detect_wildcard(
    apex_domain: str, resolver: aiodns.DNSResolver
) -> Optional[frozenset[str]]:
    """Detect whether an apex domain has a wildcard DNS record.

    Probes multiple random subdomains that are extremely unlikely to exist.
    If all probes resolve to the same set of IPs, a wildcard is present.

    Args:
        apex_domain: The apex domain to check (e.g. "example.com")
        resolver: DNS resolver instance

    Returns:
        Frozenset of wildcard IP addresses if wildcard detected, None otherwise
    """
    random_labels = [
        "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
        for _ in range(WILDCARD_PROBE_COUNT)
    ]
    probe_hosts = [f"{label}.{apex_domain}" for label in random_labels]

    ip_sets = await asyncio.gather(
        *[_resolve_ips(host, resolver) for host in probe_hosts]
    )

    # If any probe failed to resolve, there is no wildcard
    if any(len(ips) == 0 for ips in ip_sets):
        return None

    # All probes resolved — check if they share at least one common IP
    common_ips = ip_sets[0]
    for ips in ip_sets[1:]:
        common_ips = common_ips & ips

    if common_ips:
        return common_ips

    return None


async def get_registered_domains(
    domains_to_check: set[Domain],
    resolution_concurrency: int,
    apex_domain: Optional[str] = None,
) -> set[Domain]:
    """Check which domains from a set are registered and resolve to IP addresses.

    When an apex_domain is provided, performs wildcard detection first and filters
    out domains that resolve solely to wildcard IP addresses.

    Args:
        domains_to_check: Set of Domain objects to check for registration
        resolution_concurrency: Maximum number of concurrent DNS resolutions
        apex_domain: Optional apex domain for wildcard detection

    Returns:
        Set of Domain objects that are registered and resolve successfully,
        excluding wildcard false positives
    """
    semaphore = asyncio.Semaphore(resolution_concurrency)
    resolver = aiodns.DNSResolver(
        nameservers=NAME_SERVERS, timeout=TIMEOUT, tries=TRIES
    )

    # Detect wildcard DNS before resolving predictions
    wildcard_ips = None
    if apex_domain:
        wildcard_ips = await detect_wildcard(apex_domain, resolver)

    if wildcard_ips is None:
        # No wildcard — use the original fast path (just check if registered)
        domains_list = list(domains_to_check)
        tasks = [dom.is_registered(resolver, semaphore) for dom in domains_to_check]
        results = await asyncio.gather(*tasks)
        return {dom for dom, is_reg in zip(domains_list, results) if is_reg}

    # Wildcard detected — resolve each domain and keep only those with
    # at least one IP outside the wildcard set
    async def _resolves_non_wildcard(domain: Domain) -> bool:
        async with semaphore:
            ips = await _resolve_ips(str(domain), resolver)
            if not ips:
                return False
            # Keep the domain if it has any IP not in the wildcard set
            return not ips.issubset(wildcard_ips)

    domains_list = list(domains_to_check)
    tasks = [_resolves_non_wildcard(dom) for dom in domains_list]
    results = await asyncio.gather(*tasks)

    return {dom for dom, keep in zip(domains_list, results) if keep}
