from argparse import ArgumentTypeError

from subwiz.main import device_validator


def test_():
    assert device_validator("auto") in ["mps", "cuda", "cpu"]
    assert device_validator("mps") == "mps"
    assert device_validator("cuda") == "cuda"
    assert device_validator("cpu") == "cpu"

    try:
        device_validator("test")
        assert False
    except ArgumentTypeError:
        pass
