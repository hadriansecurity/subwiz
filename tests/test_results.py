from subwiz import run


def test_languages():
    results = run(
        input_domains=["german.hadrian.io", "italian.hadrian.io"],
        num_predictions=100,
        no_resolve=True,
        max_recursion=0,
    )
    print(results)
    assert "english.hadrian.io" in results


def test_numbers():
    results = run(
        input_domains=["test1.hadrian.io", "test2.hadrian.io"],
        num_predictions=10,
        no_resolve=True,
        max_recursion=0,
    )
    assert "test3.hadrian.io" in results
