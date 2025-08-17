"""Tests for DNS resolution functionality.

This module contains tests that verify the DNS resolution and domain
registration checking works correctly for various domain inputs.
"""

import asyncio

from subwiz.resolve import get_registered_domains
from subwiz.type import Domain


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
