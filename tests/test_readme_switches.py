import os
import subprocess


def test_():
    with open("README.md", "r") as f:
        readme = f.read()

    result = subprocess.run(
        ["python", "-m", "subwiz.cli", "-h"], capture_output=True, text=True
    )
    help_text = result.stdout

    assert help_text in readme
