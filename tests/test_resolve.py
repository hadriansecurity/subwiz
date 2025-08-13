import asyncio

from subwiz.resolve import get_registered_domains
from subwiz.type import Domain


def test_():
    domain_strings = {"api.hadrian.io", "app.hadrian.io", "random_string.hadrian.io"}
    input_domains = {Domain(dom) for dom in domain_strings}
    registered_domains = asyncio.run(
        get_registered_domains(input_domains, resolution_concurrency=10)
    )
    assert registered_domains == {Domain("api.hadrian.io"), Domain("app.hadrian.io")}
