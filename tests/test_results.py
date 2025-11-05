"""Tests for subdomain generation results and validation.

This module contains tests that verify the subdomain generation functionality
works correctly for different types of input patterns and produces expected results.
"""

from subwiz import run


def test_languages():
    """Test that language-based subdomain patterns are generated correctly.

    Verifies that when given language subdomains (german, italian), the model
    can generate related language subdomains like 'english'.
    """
    results = run(
        input_domains=["german.hadrian.io", "italian.hadrian.io"],
        num_predictions=100,
        no_resolve=True,
    )
    print(results)
    assert "english.hadrian.io" in results


def test_numbers():
    """Test that numeric subdomain patterns are generated correctly.

    Verifies that when given numeric subdomains (test1, test2), the model
    can generate the next numeric subdomain like 'test3'.
    """
    results = run(
        input_domains=["test1.hadrian.io", "test2.hadrian.io"],
        num_predictions=10,
        no_resolve=True,
    )
    assert "test3.hadrian.io" in results
