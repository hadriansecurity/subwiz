import pytest
import argparse
import os
import tempfile
from unittest.mock import patch, MagicMock
import torch

from subwiz.type import (
    Domain,
    max_recursion_type,
    positive_int_type,
    input_domains_file_type,
    input_domains_type,
    output_file_type,
    temperature_type,
    concurrency_type,
    device_type,
)


class TestDomain:
    """Test the Domain class with various domain examples"""

    def test_valid_domain_with_subdomain(self):
        """Test creating a domain with subdomain"""
        domain = Domain("sub.example.com")
        assert domain.subdomain == "sub"
        assert domain.apex_domain == "example.com"
        assert str(domain) == "sub.example.com"

    def test_valid_domain_without_subdomain(self):
        """Test creating a domain without subdomain"""
        domain = Domain("example.com")
        assert domain.subdomain == ""
        assert domain.apex_domain == "example.com"
        assert str(domain) == "example"

    def test_domain_with_multiple_subdomains(self):
        """Test creating a domain with multiple subdomains"""
        domain = Domain("a.b.c.example.com")
        assert domain.subdomain == "a.b.c"
        assert domain.apex_domain == "example.com"
        assert str(domain) == "a.b.c.example.com"

    def test_domain_with_tld_only(self):
        """Test creating a domain with only TLD (should fail)"""
        with pytest.raises(ValueError, match="not a valid domain"):
            Domain(".com")

    def test_domain_with_empty_string(self):
        """Test creating a domain with empty string (should fail)"""
        with pytest.raises(ValueError, match="not a valid domain"):
            Domain("")

    def test_domain_equality(self):
        """Test domain equality comparison"""
        domain1 = Domain("example.com")
        domain2 = Domain("example.com")
        domain3 = Domain("sub.example.com")

        assert domain1 == domain2
        assert domain1 != domain3
        assert hash(domain1) == hash(domain2)

    def test_domain_hash(self):
        """Test domain hashing"""
        domain1 = Domain("test.com")
        domain2 = Domain("test.com")
        assert hash(domain1) == hash(domain2)


class TestMaxRecursionType:
    """Test the max_recursion_type function with examples"""

    def test_valid_integer_string(self):
        """Test with valid integer string"""
        assert max_recursion_type("25") == 25
        assert max_recursion_type("0") == 0
        assert max_recursion_type("50") == 50

    def test_valid_integer(self):
        """Test with valid integer"""
        assert max_recursion_type(25) == 25
        assert max_recursion_type(0) == 0
        assert max_recursion_type(50) == 50

    def test_invalid_string(self):
        """Test with invalid string"""
        with pytest.raises(
            argparse.ArgumentTypeError, match="use an integer >= 0 and <= 50"
        ):
            max_recursion_type("abc")

    def test_negative_integer(self):
        """Test with negative integer"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use an integer >= 0 and <= 50",
        ):
            max_recursion_type(-1)

    def test_too_large_integer(self):
        """Test with integer > 50"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use an integer >= 0 and <= 50",
        ):
            max_recursion_type(51)


class TestPositiveIntType:
    """Test the positive_int_type function with examples"""

    def test_valid_positive_integer_string(self):
        """Test with valid positive integer string"""
        assert positive_int_type("25") == 25
        assert positive_int_type("1") == 1

    def test_valid_positive_integer(self):
        """Test with valid positive integer"""
        assert positive_int_type(25) == 25
        assert positive_int_type(1) == 1

    def test_invalid_string(self):
        """Test with invalid string"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use a positive integer",
        ):
            positive_int_type("abc")

    def test_zero(self):
        """Test with zero"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use a positive integer",
        ):
            positive_int_type(0)

    def test_negative_integer(self):
        """Test with negative integer"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use a positive integer",
        ):
            positive_int_type(-1)


class TestInputDomainsType:
    """Test the input_domains_type function with examples"""

    def test_valid_domains_with_subdomain(self):
        """Test with valid domains including subdomain"""
        domains = ["sub.example.com", "example.com", "test.example.org"]
        result = input_domains_type(domains)
        assert len(result) == 3
        assert all(isinstance(d, Domain) for d in result)

    def test_empty_list(self):
        """Test with empty list"""
        with pytest.raises(
            argparse.ArgumentTypeError, match="empty input domains"
        ):
            input_domains_type([])

    def test_no_subdomains(self):
        """Test with domains but no subdomains"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="input should include at least one subdomain",
        ):
            input_domains_type(["example.com", "test.org"])

    def test_invalid_domains(self):
        """Test with invalid domains"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="invalid input domains",
        ):
            input_domains_type(["invalid", "sub.example.com"])

    def test_duplicate_domains(self):
        """Test with duplicate domains"""
        domains = ["sub.example.com", "sub.example.com", "test.example.org"]
        result = input_domains_type(domains)
        assert len(result) == 2  # Duplicates should be removed


class TestInputDomainsFileType:
    """Test the input_domains_file_type function with examples"""

    def test_valid_file_with_domains(self):
        """Test with valid file containing domains"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("sub.example.com\nexample.com\ntest.example.org")
            f.flush()
            
            try:
                result = input_domains_file_type(f.name)
                assert len(result) == 3
                assert all(isinstance(d, Domain) for d in result)
            finally:
                os.unlink(f.name)

    def test_file_not_found(self):
        """Test with non-existent file"""
        with pytest.raises(
            argparse.ArgumentTypeError, match="file not found"
        ):
            input_domains_file_type("/nonexistent/file.txt")

    def test_invalid_input_type(self):
        """Test with invalid input type"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use a pathlike input for file path",
        ):
            input_domains_file_type(123)


class TestOutputFileType:
    """Test the output_file_type function with examples"""

    def test_none_value(self):
        """Test with None value"""
        assert output_file_type(None) is None

    def test_valid_string_path(self):
        """Test with valid string path"""
        result = output_file_type("/tmp/test.txt")
        assert result == "/tmp/test.txt"

    def test_invalid_type(self):
        """Test with invalid type"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use a pathlike input for file path",
        ):
            output_file_type(123)


class TestTemperatureType:
    """Test the temperature_type function with examples"""

    def test_valid_temperature_string(self):
        """Test with valid temperature string"""
        assert temperature_type("0.5") == 0.5
        assert temperature_type("0.0") == 0.0
        assert temperature_type("1.0") == 1.0

    def test_valid_temperature_float(self):
        """Test with valid temperature float"""
        assert temperature_type(0.5) == 0.5
        assert temperature_type(0.0) == 0.0
        assert temperature_type(1.0) == 1.0

    def test_invalid_string(self):
        """Test with invalid string"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="not a valid float",
        ):
            temperature_type("abc")

    def test_temperature_below_zero(self):
        """Test with temperature below 0"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use 0 ≤ temperature ≤ 1",
        ):
            temperature_type(-0.1)

    def test_temperature_above_one(self):
        """Test with temperature above 1"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use 0 ≤ temperature ≤ 1",
        ):
            temperature_type(1.1)


class TestConcurrencyType:
    """Test the concurrency_type function with examples"""

    def test_valid_concurrency_string(self):
        """Test with valid concurrency string"""
        assert concurrency_type("10") == 10
        assert concurrency_type("1") == 1
        assert concurrency_type("256") == 256

    def test_valid_concurrency_int(self):
        """Test with valid concurrency int"""
        assert concurrency_type(10) == 10
        assert concurrency_type(1) == 1
        assert concurrency_type(256) == 256

    def test_invalid_string(self):
        """Test with invalid string"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="not a valid int",
        ):
            concurrency_type("abc")

    def test_concurrency_below_one(self):
        """Test with concurrency below 1"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use 1 ≤ concurrency ≤ 256",
        ):
            concurrency_type(0)

    def test_concurrency_above_256(self):
        """Test with concurrency above 256"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use 1 ≤ concurrency ≤ 256",
        ):
            concurrency_type(257)


class TestDeviceType:
    """Test the device_type function with examples"""

    def test_valid_device_strings(self):
        """Test with valid device strings"""
        assert device_type("cpu") == "cpu"
        assert device_type("cuda") == "cuda"
        assert device_type("mps") == "mps"

    def test_invalid_string(self):
        """Test with invalid string"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match="use a string for the device",
        ):
            device_type(123)

    def test_invalid_device_name(self):
        """Test with invalid device name"""
        with pytest.raises(
            argparse.ArgumentTypeError,
            match='device should be in \["auto", "cpu", "cuda", "mps"\]',
        ):
            device_type("invalid")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_device_cuda_available(self, mock_mps, mock_cuda):
        """Test auto device when CUDA is available"""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        assert device_type("auto") == "cuda"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_device_mps_available(self, mock_mps, mock_cuda):
        """Test auto device when MPS is available but CUDA is not"""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        assert device_type("auto") == "mps"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_device_cpu_fallback(self, mock_mps, mock_cuda):
        """Test auto device falls back to CPU when neither CUDA nor MPS is available"""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        assert device_type("auto") == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
