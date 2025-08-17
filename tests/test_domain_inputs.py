"""Tests for domain input validation and file processing.

This module contains tests that verify the domain input validation functionality,
including file reading, domain format validation, and multi-apex domain handling.
"""

import argparse
import os

from subwiz.type import input_domains_file_type
from subwiz.main import run


def test_file_doesnt_exist():
    """Test that non-existent files raise appropriate errors.

    Verifies that attempting to read from a non-existent file raises
    an ArgumentTypeError.
    """
    try:
        input_domains_file_type("nonexistent_file.txt")
        assert False
    except argparse.ArgumentTypeError:
        pass


def test_good_file():
    """Test that valid domain files are processed correctly.

    Creates a temporary test file with valid domains and verifies that
    the input_domains_file_type function can read and process it without errors.
    """
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("admin.hadrian.io\ntest.hadrian.io")
    input_domains_file_type(test_file_name)
    os.remove(test_file_name)


def test_empty_file():
    """Test that empty files raise appropriate errors.

    Verifies that attempting to read from an empty file raises
    an ArgumentTypeError.
    """
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
    """Test that files with invalid domain formats raise errors.

    Verifies that attempting to read from a file containing invalid
    domain strings raises an ArgumentTypeError.
    """
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
    """Test multi-apex domain handling with and without the multi-apex flag.

    Verifies that running with multiple apex domains fails without the
    --multi-apex flag and succeeds with it. Tests both error conditions
    and successful execution paths.
    """
    test_file_name = ".test_multi_apex.txt"
    multi_apex_domains = [
        "wfs.preprod.onmicrosoft.com",
        "loki-elk.staging.msft.ai",
        "fax-and-scan.staging.onmicrosoft.com",
    ]

    with open(test_file_name, "w") as f:
        f.write("\n".join(multi_apex_domains))

    try:
        run(input_domains=multi_apex_domains, no_resolve=True, multi_apex=False)
        assert False, "Should have failed without --multi-apex flag"
    except argparse.ArgumentTypeError:
        pass

    try:
        results = run(
            input_domains=multi_apex_domains,
            no_resolve=True,
            multi_apex=True,
            num_predictions=5,
        )
        assert isinstance(results, list)
    except Exception as e:
        assert False, f"Multi-apex run failed with an exception: {e}"

    os.remove(test_file_name)
