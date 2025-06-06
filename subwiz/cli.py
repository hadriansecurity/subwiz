# ==> subwiz/cli.py <==
"""Exposes a command line interface that runs subwiz and returns the result."""

import argparse
from collections import defaultdict

from subwiz.main import (
    download_files,
    run_inference,
    run_resolution,
)
from subwiz.type import (
    Domain,
    device_type,
    input_domains_file_type,
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
    "--num_predictions",
    help="number of subdomains to predict.",
    dest="num_predictions",
    default=500,
    type=positive_int_type,
)
parser.add_argument(
    "--no-resolve",
    help="do not resolve the output subdomains. ",
    dest="no_resolve",
    action="store_true",
)
parser.add_argument(
    "--force-download",
    help="download model and tokenizer files, even if cached. ",
    dest="force_download",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--temperature",
    help="add randomness to the model, recommended ≤ 0.3)",
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
    "-q",
    "--max_new_tokens",
    help="maximum length of predicted subdomains in tokens.",
    dest="max_new_tokens",
    default=10,
    type=positive_int_type,
)
parser.add_argument(
    "--resolution_concurrency",
    help="number of concurrent resolutions.",
    dest="resolution_lim",
    default=128,
    type=positive_int_type,
)
parser.add_argument(
    "--multi-apex",
    help="allow multiple apex domains in the input file. runs inference for each apex separately.",
    dest="multi_apex",
    action="store_true",
)
args = parser.parse_args()


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


hello_message = """
███████╗██╗   ██╗██████╗     ██╗   ██╗██╗███████╗
██╔════╝██║   ██║██╔══██╗    ██║   ██║██║╚══███╔╝
███████╗██║   ██║██████╔╝    ██║ █╗ ██║██║  ███╔╝ 
╚════██║██║   ██║██╔══██╗    ██║███╗██║██║ ███╔╝  
███████║╚██████╔╝██████╔╝    ╚███╔███╔╝██║███████╗
╚══════╝ ╚═════╝ ╚═════╝      ╚══╝╚══╝ ╚═╝╚══════╝"""


def print_hello():
    print(f"{bcolors.OKGREEN}{hello_message}{bcolors.ENDC}", flush=True)


def print_log(msg: str, end="\n"):
    print(f"{bcolors.OKCYAN}{msg}{bcolors.ENDC}", flush=True, end=end)


def print_progress_dot():
    print_log(".", end="")


def main():
    print_hello()

    # 1. Group the input domains by their apex
    domain_objects: list[Domain] = args.input_file
    domain_groups = defaultdict(list)
    for dom in domain_objects:
        domain_groups[dom.apex_domain].append(dom)

    # 2. Check for multiple apex domains IF the flag isn't set
    if len(domain_groups) > 1 and not args.multi_apex:
        parser.error(
            f"multiple apex domains found: {sorted(domain_groups.keys())}. "
            "Use the --multi-apex flag to process them all."
        )

    # 3. Download files once
    model_path, tokenizer_path = download_files(force_download=args.force_download)
    all_predictions = set()

    # 4. Loop through each group and run inference
    for apex, domains_in_group in domain_groups.items():
        print_log(f"[*] running inference for {apex}", end="")

        predictions = run_inference(
            input_domains=domains_in_group,
            device=args.device,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            num_predictions=args.num_predictions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            on_inference_iteration=print_progress_dot,
        )
        print_log("")  # end line after progress dots
        all_predictions.update(predictions)

    final_predictions = sorted(list(all_predictions))

    if not args.no_resolve:
        print_log("resolving subdomains...")
        final_predictions = run_resolution(final_predictions, resolution_lim=args.resolution_lim)

    output = "\n".join(sorted(final_predictions))

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()

