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
███████╗██╗   ██╗██████╗     ██╗    ██╗██╗███████╗
██╔════╝██║   ██║██╔══██╗    ██║    ██║██║╚══███╔╝
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
