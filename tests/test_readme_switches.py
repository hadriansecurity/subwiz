import subprocess


def test_():
    with open("README.md", "r") as f:
        readme = f.read()

    result = subprocess.run(
        ["python", "-m", "subwiz.cli", "-h"], capture_output=True, text=True, env={**dict(subprocess.os.environ), "COLUMNS": "100"}
    )
    help_text = result.stdout

    print(help_text)

    assert help_text in readme
