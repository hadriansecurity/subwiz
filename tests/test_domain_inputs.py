import argparse
import os

from subwiz.type import input_domains_file_type
from subwiz.main import run


def test_file_doesnt_exist():
    try:
        input_domains_file_type("nonexistent_file.txt")
        assert False
    except argparse.ArgumentTypeError:
        pass


def test_good_file():
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("admin.hadrian.io\ntest.hadrian.io")
    input_domains_file_type(test_file_name)
    os.remove(test_file_name)


def test_empty_file():
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("")
    try:
        input_domains_file_type(test_file_name)
        os.remove(test_file_name)
        assert False
    except argparse.ArgumentTypeError:
        os.remove(test_file_name)
        pass


def test_non_domain_file():
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("testing\ntesting")
    try:
        input_domains_file_type(test_file_name)
        os.remove(test_file_name)
        assert False
    except argparse.ArgumentTypeError:
        os.remove(test_file_name)
        pass


def test_multi_apex():
    test_file_name = ".test_multi_apex.txt"
    multi_apex_domains = ["wfs.preprod.onmicrosoft.com", "loki-elk.staging.msft.ai", "fax-and-scan.staging.onmicrosoft.com"]
    with open(test_file_name, "w") as f:
        f.write("\n".join(multi_apex_domains))

    try:
        run(input_domains=multi_apex_domains, no_resolve=True, multi_apex=False)
        assert False, "Should have failed without --multi-apex flag"
    except argparse.ArgumentTypeError:
        pass

    try:
        results = run(input_domains=multi_apex_domains, no_resolve=True, multi_apex=True, num_predictions=5)
        assert isinstance(results, list)
    except Exception as e:
        assert False, f"Multi-apex run failed with an exception: {e}"

    os.remove(test_file_name)
