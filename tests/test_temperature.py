"""Tests for temperature parameter behavior in subdomain generation.

This module contains tests that verify the temperature parameter correctly
controls the randomness and determinism of subdomain generation results.
"""

from argparse import ArgumentTypeError

from subwiz import run


def test_zero():
    """Test that temperature=0 produces deterministic results.

    Verifies that running the same input with temperature=0 produces
    identical results on multiple runs, ensuring deterministic behavior.
    """
    args = {
        "input_domains": ["admin.hadrian.io", "test.hadrian.io"],
        "num_predictions": 10,
        "temperature": 0,
    }
    first_results = run(**args, no_resolve=True)
    second_results = run(**args, no_resolve=True)

    assert first_results == second_results


def test_high():
    """Test that temperature=1 produces non-deterministic results.

    Verifies that running the same input with temperature=1 produces
    different results on multiple runs, ensuring non-deterministic behavior.
    """
    args = {
        "input_domains": ["admin.hadrian.io", "test.hadrian.io"],
        "num_predictions": 20,
        "temperature": 1,
    }
    first_results = run(**args, no_resolve=True)
    second_results = run(**args, no_resolve=True)

    assert first_results != second_results


def test_out_of_bounds():
    """Test that out-of-bounds temperature values raise errors.

    Verifies that temperature values outside the valid range [0, 1]
    raise ArgumentTypeError for both upper and lower bounds.
    """
    try:
        run(
            input_domains=["admin.hadrian.io"],
            temperature=1.1,
            num_predictions=1,
            no_resolve=True,
        )
        assert False
    except ArgumentTypeError:
        pass

    try:
        run(
            input_domains=["admin.hadrian.io"],
            temperature=-0.1,
            num_predictions=1,
            no_resolve=True,
        )
        assert False
    except ArgumentTypeError:
        pass
