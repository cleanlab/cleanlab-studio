## Development

1. To install the package locally for development, clone this repo and run `pip install --editable .` from the home
   directory. Changes to the code are reflected automatically in the CLI.
2. `Makefile` contains sample commands for quick installation and testing, though you will have to specify filepaths and
   API keys manually.

## Formatting

Cleanlab CLI uses [Black](https://black.readthedocs.io/en/stable/) to standardize code formatting. Black is configured
with `pyproject.toml`.

Developers should also set up pre-commit hooks to re-format any changed code prior to a commit. The configuration for
pre-commit is in `pre-commit-config.yaml` and `pre-commit-hooks.yaml`.

On every push to the repo, GitHub Actions checks for formatting issues using Black as well. The configuration options
are in `.github/workflows/format.yml`.

### Setup instructions

To set up pre-commit:

```
python -m pip install -r requirements-dev.txt
pre-commit install
```

To run the formatter manually:

```
black .
```

## Uploading to PyPI

References:

- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://twine.readthedocs.io/en/stable/

1. python3 -m pip install --upgrade build
2. python3 -m build
3. python3 -m pip install --upgrade twine

To upload to TestPyPi:

4. twine upload -r testpypi dist/*
5. pip install -i https://test.pypi.org/simple/ cleanlab-cli

This last step may fail if test versions of some required packages are not available.

To upload to PyPi:

6. twine upload dist/*
