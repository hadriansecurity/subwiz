from argparse import ArgumentTypeError

from subwiz.main import device_type


def test_():
    assert device_type("auto") in ["mps", "cuda", "cpu"]
    assert device_type("mps") == "mps"
    assert device_type("cuda") == "cuda"
    assert device_type("cpu") == "cpu"

    try:
        device_type("test")
        assert False
    except ArgumentTypeError:
        pass
