"""Tests for DNS resolution functionality.

This module contains tests that verify the DNS resolution and domain
registration checking works correctly for various domain inputs,
including wildcard DNS detection and filtering.
"""

import asyncio

import aiodns

from subwiz.resolve import detect_wildcard, get_registered_domains, NAME_SERVERS, TIMEOUT, TRIES
from subwiz.type import Domain


def test_registered_domains():
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


def test_wildcard_detection_non_wildcard():
    """Test that wildcard detection returns None for non-wildcard domains."""

    async def _check():
        resolver = aiodns.DNSResolver(
            nameservers=NAME_SERVERS, timeout=TIMEOUT, tries=TRIES
        )
        return await detect_wildcard("hadrian.io", resolver)

    # hadrian.io does not have a wildcard record
    result = asyncio.run(_check())
    assert result is None


def test_registered_domains_with_apex():
    """Test that passing apex_domain still returns correct results for non-wildcard domains."""
    domain_strings = {"api.hadrian.io", "app.hadrian.io", "random_string.hadrian.io"}
    input_domains = {Domain(dom) for dom in domain_strings}
    registered_domains = asyncio.run(
        get_registered_domains(
            input_domains, resolution_concurrency=10, apex_domain="hadrian.io"
        )
    )
    assert registered_domains == {Domain("api.hadrian.io"), Domain("app.hadrian.io")}
