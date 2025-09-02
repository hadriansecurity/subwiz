"""Tests for device type validation and auto-detection.

This module contains tests that verify the device type validation
functionality works correctly for various device inputs and auto-detection.
"""

from argparse import ArgumentTypeError

from subwiz.main import device_type


def test_():
    """Test device type validation and auto-detection functionality.

    Verifies that valid device types are accepted, invalid device types
    raise errors, and the auto-detection correctly identifies available
    hardware (MPS, CUDA, or CPU).
    """
    assert device_type("auto") in ["mps", "cuda", "cpu"]
    assert device_type("mps") == "mps"
    assert device_type("cuda") == "cuda"
    assert device_type("cpu") == "cpu"

    try:
        device_type("test")
        assert False
    except ArgumentTypeError:
        pass
