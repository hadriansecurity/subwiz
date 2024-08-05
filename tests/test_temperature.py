from argparse import ArgumentTypeError

from subwiz import run


def test_zero():
    args = {
        "input_domains": ["admin.hadrian.io", "test.hadrian.io"],
        "num_predictions": 10,
        "temperature": 0,
    }
    first_results = run(**args, no_resolve=True)
    second_results = run(**args, no_resolve=True)

    assert first_results == second_results


def test_high():
    args = {
        "input_domains": ["admin.hadrian.io", "test.hadrian.io"],
        "num_predictions": 20,
        "temperature": 1,
    }
    first_results = run(**args, no_resolve=True)
    second_results = run(**args, no_resolve=True)

    assert first_results != second_results


def test_out_of_bounds():
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
