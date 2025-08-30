"""Exposes a command line interface that runs subwiz and returns the result."""

import argparse

from subwiz.main import run
from subwiz.type import (
    Domain,
    device_type,
    input_domains_file_type,
    max_recursion_type,
    output_file_type,
    positive_int_type,
    temperature_type,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input-file",
    help="file containing new-line-separated subdomains.",
    dest="input_file",
    required=True,
    type=input_domains_file_type,
)
parser.add_argument(
    "-o",
    "--output-file",
    help="output file to write new-line separated subdomains to.",
    dest="output_file",
    type=output_file_type,
)
parser.add_argument(
    "-n",
    "--num-predictions",
    help="number of subdomains to predict.",
    dest="num_predictions",
    default=500,
    type=positive_int_type,
)
parser.add_argument(
    "--no-resolve",
    help="do not resolve the output subdomains.",
    dest="no_resolve",
    action="store_true",
)
parser.add_argument(
    "--force-download",
    help="download model and tokenizer files, even if cached.",
    dest="force_download",
    action="store_true",
)
parser.add_argument(
    "--max-recursion",
    help="maximum number of times the inference process will recursively re-run after finding new subdomains.",
    dest="max_recursion",
    default=5,
    type=max_recursion_type,
)
parser.add_argument(
    "-t",
    "--temperature",
    help="add randomness to the model (recommended ≤ 0.3).",
    dest="temperature",
    default=0.0,
    type=temperature_type,
)
parser.add_argument(
    "-d",
    "--device",
    help="hardware to run the transformer model on.",
    dest="device",
    default="auto",
    choices=["auto", "cpu", "cuda", "mps"],
    type=device_type,
)
parser.add_argument(
    "-m",
    "--max-new-tokens",
    help="maximum length of predicted subdomains in tokens.",
    dest="max_new_tokens",
    default=10,
    type=positive_int_type,
)
parser.add_argument(
    "--resolution-concurrency",
    help="number of concurrent resolutions.",
    dest="resolution_concurrency",
    default=128,
    type=positive_int_type,
)
parser.add_argument(
    "--multi-apex",
    help="allow multiple apex domains in the input file. runs inference for each apex separately.",
    dest="multi_apex",
    action="store_true",
)
parser.add_argument(
    "-q",
    "--quiet",
    help="useful for piping into another tool.",
    action="store_true",
)
parser.add_argument(
    "-s",
    "--silent",
    help="do not print any output. requires --output-file.",
    action="store_true",
)
args = parser.parse_args()


def main():
    try:
        if args.silent and not args.output_file:
            parser.error("--silent requires --output-file.")

        domain_objects: list[Domain] = args.input_file
        input_domains = [str(dom) for dom in domain_objects]

        run_args = {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in {
                "input_file",
                "output_file",
                "profile",
                "profile_output",
                "silent",
                "quiet",
            }
        }

        results = run(
            **run_args,
            input_domains=input_domains,
            quiet=args.quiet or args.silent,
        )

        output = "\n".join(sorted(results))

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(output)

        if not args.silent:
            print(output)

    except argparse.ArgumentTypeError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
