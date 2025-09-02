"""Command-line interface printing utilities for subwiz.

This module provides colored output and formatted printing functions for the CLI,
including ASCII art banners, progress indicators, and status messages.
"""


class bcolors:
    """ANSI color codes for terminal output formatting."""

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
███████╗██╗   ██╗██████╗     ██╗    ██╗██╗███████╗
██╔════╝██║   ██║██╔══██╗    ██║    ██║██║╚══███╔╝
███████╗██║   ██║██████╔╝    ██║ █╗ ██║██║  ███╔╝ 
╚════██║██║   ██║██╔══██╗    ██║███╗██║██║ ███╔╝  
███████║╚██████╔╝██████╔╝    ╚███╔███╔╝██║███████╗
╚══════╝ ╚═════╝ ╚═════╝      ╚══╝╚══╝ ╚═╝╚══════╝"""


def print_hello():
    """Print the subwiz ASCII art banner in green color."""
    print(f"{bcolors.OKGREEN}{hello_message}{bcolors.ENDC}", flush=True)


def print_log(msg: str, end="\n"):
    """Print a log message in cyan color.

    Args:
        msg: Message to print
        end: String to append at the end (default: newline)
    """
    print(f"{bcolors.OKCYAN}{msg}{bcolors.ENDC}", flush=True, end=end)


def print_progress_dot():
    """Print a progress dot in cyan color without newline."""
    print_log(".", end="")
