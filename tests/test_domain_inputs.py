from argparse import ArgumentTypeError
import os

from subwiz.main import input_domains_file_validator


def test_file_doesnt_exist():
    try:
        input_domains_file_validator("nonexistent_file.txt")
        assert False
    except ArgumentTypeError:
        pass


def test_good_file():
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("admin.hadrian.io\ntest.hadrian.io")
    input_domains_file_validator(test_file_name)
    os.remove(test_file_name)


def test_empty_file():
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("")
    try:
        input_domains_file_validator(test_file_name)
        os.remove(test_file_name)
        assert False
    except ArgumentTypeError:
        os.remove(test_file_name)
        pass


def test_non_domain_file():
    test_file_name = ".test_input_domains_file.txt"
    with open(test_file_name, "w") as f:
        f.write("testing\ntesting")
    try:
        input_domains_file_validator(test_file_name)
        os.remove(test_file_name)
        assert False
    except ArgumentTypeError:
        os.remove(test_file_name)
        pass
