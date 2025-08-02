from subwiz.main import run_resolution


def test_():
    input_domains = {"api.hadrian.io", "random_test.hadrian.io", "app.hadrian.io"}
    registered_domains = run_resolution(input_domains, resolution_concurrency=10)
    assert registered_domains == {"api.hadrian.io", "app.hadrian.io"}
