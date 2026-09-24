"""Tests for DNS resolution functionality.

This module contains tests that verify the DNS resolution and domain
registration checking works correctly for various domain inputs, including
wildcard detection and wildcard-aware filtering.
"""

import asyncio

from subwiz.resolve import (
    NAME_SERVERS,
    TIMEOUT,
    TRIES,
    _resolve_ips,
    detect_wildcard,
    get_registered_domains,
)
from subwiz.type import Domain

import aiodns


def test_():
    """Test that DNS resolution correctly identifies registered domains.

    Verifies that the get_registered_domains function can distinguish between
    registered domains (api.hadrian.io, app.hadrian.io) and unregistered
    domains (random_string.hadrian.io).
    """
    domain_strings = {"api.hadrian.io", "app.hadrian.io", "random_string.hadrian.io"}
    input_domains = {Domain(dom) for dom in domain_strings}
    registered_domains = asyncio.run(
        get_registered_domains(input_domains, resolution_concurrency=10)
    )
    assert registered_domains == {Domain("api.hadrian.io"), Domain("app.hadrian.io")}


def test_detect_wildcard_non_wildcard():
    """A domain without a wildcard record returns an empty catch-all set."""
    wildcard_ips = asyncio.run(detect_wildcard("hadrian.io"))
    assert wildcard_ips == set()


def test_get_registered_domains_wildcard_keeps_real():
    """Real subdomains resolving outside the wildcard set are kept.

    Uses a TEST-NET address (RFC 5737, never routed to a real host) as the
    wildcard set, so any real subdomain resolves outside it and is kept.
    """
    domain_strings = {"api.hadrian.io", "app.hadrian.io", "random_string.hadrian.io"}
    input_domains = {Domain(dom) for dom in domain_strings}
    registered_domains = asyncio.run(
        get_registered_domains(
            input_domains, resolution_concurrency=10, wildcard_ips={"192.0.2.1"}
        )
    )
    assert registered_domains == {Domain("api.hadrian.io"), Domain("app.hadrian.io")}


def test_get_registered_domains_wildcard_filters_catch_all():
    """A subdomain resolving only to wildcard IPs is filtered out.

    Learns api.hadrian.io's real IPs, then treats those exact IPs as the
    wildcard set so the domain has no IP outside it and is dropped.
    """
    api = Domain("api.hadrian.io")

    async def _run() -> set[Domain]:
        resolver = aiodns.DNSResolver(
            nameservers=NAME_SERVERS, timeout=TIMEOUT, tries=TRIES
        )
        semaphore = asyncio.Semaphore(1)
        real_ips = await _resolve_ips(str(api), resolver, semaphore)
        assert real_ips, "api.hadrian.io should resolve"
        return await get_registered_domains(
            {api}, resolution_concurrency=10, wildcard_ips=real_ips
        )

    assert asyncio.run(_run()) == set()
