Thanks for considering contributing to subwiz! Here are some instructions on how to get started:

## How to contribute

- Open an issue to report a bug, suggest a new feature or an enhancement of an existing feature.
- Improve the documentation by expanding `README.md`, or adding docstrings to functions.
- Submit pull requests to fix open issues or new ones.
- Write new tests.


## How to open a PR

1. Fork this repository
2. Clone your fork locally
    ```bash
    git clone https://github.com/YOUR_USERNAME/subwiz.git
    cd subwiz
    ```
3. Set up a virtual environment
    ```
    python -m venv .venv
    .venv/bin/activate (or .venv\Scripts\activate on Windows)
    ```
4. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
5. Make a new branch
    ```bash
    git checkout -b your-branch-name
    ```
6. Make all changes.
7. Before pushing changes

   a. Run tests
    ```bash
    pip install pytest
    pytest
    ```
   b. Format code with black
    ```bash
    pip install black
    black ./subwiz --color
    black ./tests --color
    ```
8. Push changes to your fork, and open the PR! In the PR description include a summary of your changes and
   how they can be tested locally. If the PR fixes an open issue, include a link.

## Code Conventions

- Separation of concerns into single files in the subwiz package (e.g. all cli related code in `cli.py`)
- Use type hints for all functions. While mypy isn't part of CI yet (as not all mypy errors have been resolved) 
  it's a good idea to run mypy to check that new errors haven't crept in.
- For new functionality include tests.
- If arguments or functionality is changed, please update `README.md`.
