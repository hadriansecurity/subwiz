from setuptools import setup, find_packages

setup(
    name="subwiz",
    version="0.1.5",
    description="A recon tool that uses AI to predict subdomains. Then returns those that resolve.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Klaas Meinke",
    author_email="klaas@hadrian.io",
    python_requires=">=3.9, <3.13",
    license="MIT License",
    keywords=["machine learning", "recon", "subdomains", "transformers"],
    url="https://github.com/hadriansecurity/subwiz",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "aiodns>=1.0.0",
        "huggingface-hub>=0.5.0",
        "pydantic>=1.10.0",
        "tldextract>=3.0.0",
        "torch>=2.3.0",
        "transformers>=4.34.0",
    ],
    entry_points={
        "console_scripts": [
            "subwiz=subwiz.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
    ],
)
