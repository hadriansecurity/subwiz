from setuptools import setup, find_packages

setup(
    name='subwiz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "aiodns>=1.0.0",
        "huggingface-hub>=0.5.0",
        "pydantic>=1.10.0",
        "tldextract>=3.0.0",
        "torch>=2.3.0",
        "transformers>=4.34.0"
    ],
    entry_points={
        'console_scripts': [
            'subwiz = subwiz:cli.main'
        ]
    }
)
