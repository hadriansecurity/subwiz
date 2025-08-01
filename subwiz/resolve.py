import asyncio

import aiodns
import idna.core


NAME_SERVERS = ["1.1.1.1", "1.0.0.1", "8.8.8.8"]
TIMEOUT = 3
TRIES = 1
DNS_RECORD = "A"


async def is_registered(
    domain: str, resolver: aiodns.DNSResolver, semaphore: asyncio.Semaphore
) -> bool:
    async with semaphore:
        try:
            await resolver.query(domain, DNS_RECORD)
            return True
        except idna.IDNAError:
            return False
        except aiodns.error.DNSError:
            return False


async def is_registered_bulk(
    domains_to_check: set[str], resolution_concurrency: int
) -> set[str]:
    semaphore = asyncio.Semaphore(resolution_concurrency)
    resolver = aiodns.DNSResolver(
        nameservers=NAME_SERVERS, timeout=TIMEOUT, tries=TRIES
    )

    tasks = [is_registered(dom, resolver, semaphore) for dom in domains_to_check]
    results = await asyncio.gather(*tasks)
    registered_domains = {
        dom for dom, is_reg in zip(domains_to_check, results) if is_reg
    }

    return registered_domains
