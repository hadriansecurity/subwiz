import subprocess


def standardize_text(text):
    text = text.replace(" ", "")
    text = text.replace("\t", "")
    text = text.replace("\n", "")
    return text


def test_():
    with open("README.md", "r") as f:
        readme = f.read()

    result = subprocess.run(
        ["python", "-m", "subwiz.cli", "-h"], capture_output=True, text=True
    )
    help_text = result.stdout

    # Sandardize texts
    help_text = standardize_text(help_text)
    readme = standardize_text(readme)

    assert help_text in readme
