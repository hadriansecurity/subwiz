from subwiz import run
from subwiz.type import Domain
import subprocess
import tempfile


def test_quiet_output_with_force_download():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("a.hadrian.io\nb.hadrian.io")
        input_file = f.name

    command = [
        "python",
        "-m",
        "subwiz.cli",
        "-i",
        input_file,
        "--no-resolve",
        "-n",
        "10",
        "-q",
        "--force-download",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    # Check that we still get domain output
    lines = result.stdout.strip().splitlines()
    assert lines, "No domains found in output"
    for line in lines:
        Domain(line)


def test_run_is_quiet_by_default(capsys):
    results = run(
        input_domains=["test.hadrian.io"],
        num_predictions=1,
        no_resolve=True,
    )
    assert results
    captured = capsys.readouterr()
    assert captured.out == ""


def test_silent_output():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("a.hadrian.io\nb.hadrian.io")
        input_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as out_f:
        output_file = out_f.name

    result = subprocess.run(
        [
            "python",
            "-m",
            "subwiz.cli",
            "-i",
            input_file,
            "--no-resolve",
            "-n",
            "10",
            "-s",
            "-o",
            output_file,
        ],
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_silent_without_output_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("a.hadrian.io\nb.hadrian.io")
        input_file = f.name

    result = subprocess.run(
        [
            "python",
            "-m",
            "subwiz.cli",
            "-i",
            input_file,
            "--no-resolve",
            "-n",
            "10",
            "-s",
        ],
        capture_output=True,
        text=True,
    )

    assert "error: --silent requires --output-file." in result.stderr
