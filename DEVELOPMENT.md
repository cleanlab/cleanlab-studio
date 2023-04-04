## View performance benchmarks

https://cleanlab.github.io/cleanlab-studio/dev/bench/

## Development

1. To install the package locally for development, clone this repo and run `pip install --editable .` from the home
   directory. Changes to the code are reflected automatically in the CLI.
2. `Makefile` contains sample commands for quick installation and testing, though you will have to specify filepaths and
   API keys manually.
3. Run `export CLEANLAB_API_BASE_URL="http://localhost:8500/api/cli/v0"` and `export CLEANLAB_API_BASE_URL_V1="http://localhost:8500/api/cli/v1"` so that API requests are made on your
   local machine

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

### Through GitHub Actions

- Update the version number in the repo (see **Incrementing the package version number**)
- Commit the changes in a commit titled with the new version number `v0.[x].[y]` (substitute `x` and `y`).
- **Tag** the commit with the new version number.
- Push the commit to `main`

The tagging of the commit should trigger the GitHub Actions workflow to build and release the package.

### Manually (don't do this unless the previous section fails!)

References:

- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://twine.readthedocs.io/en/stable/

1. python3 -m pip install --upgrade build
2. python3 -m pip install --upgrade twine
3. rm -rf dist (if present)
4. python3 -m build

To upload to TestPyPi:

1. twine upload -r testpypi dist/\*
2. pip install -i https://test.pypi.org/simple/ cleanlab-studio

This last step may fail if test versions of some required packages are not available.

To upload to PyPi:

1. twine upload dist/\*

## Versioning

For _users_, there is only one version number to keep track of: the CLI package version.

For _developers_, there are four version numbers to keep track of:

1. The CLI package version
2. Cleanlab Studio CLI API version (currently v0)
3. The schema version number
4. The CLI settings version number

The latest version numbers for (1), (2), and (4)
are [stored in version.py](https://github.com/cleanlab/cleanlab-studio/blob/main/cleanlab_studio/version.py).

### Minimum supported versions

Each **version of the CLI** supports some:

- `MIN_SCHEMA_VERSION`: Minimum schema version number
- `MIN_SETTINGS_VERSION`: Minimum CLI settings version number

If a user provides a schema with version number < `MIN_SCHEMA_VERSION`, it cannot be used. A new one must be generated.

If a CLI settings file has a version number < `MIN_SETTINGS_VERSION`, it cannot be used. The CLI will attempt to migrate
it (with the user's permission).

Similarly, each version of the **Cleanlab Studio CLI API** supports some:

- `MIN_CLI_VERSION`: Minimum CLI version number

The CLI, upon initializing, pings the CLI API with its version number to check if it is compatible. If the CLI
version < `MIN_CLI_VERSION`, then the user is prompted to upgrade their `cleanlab-studio` package.

Each **version of the CLI** also supports some:

- `MAX_SCHEMA_VERSION`: Maximum schema version number
- `MAX_SETTINGS_VERSION`: Maximum CLI settings version number

These are the maximum versions for the schema / settings that the CLI is able to handle. Every time the schema /
settings version is incremented, the `MAX_SCHEMA_VERSION` / `MAX_SETTINGS_VERSION` should be updated as well.

### When to increment the minimum supported versions

For each **release of the CLI**, update the minimum supported versions whenever there is a change in how the CLI
interfaces with the settings or schema.

#### Examples of when to update

- The CLI now expects the Settings / Schema to have a new key, so older settings / schema would be incompatible with the
  new CLI.

#### Examples of when not to update

- CLI does additional checks on schema that it did not do before, but the schema format is unchanged
- CLI adds new functionality for interfacing with Cleanlab Studio CLI API, but no new behavior is introduced for
  interfacing with settings / schema

---

Whenever the **CLI API is updated**, update the minimum supported CLI version when there is a change in the API, which
changes the interface between API and CLI in a way that breaks compatibility. Every endpoint in the CLI API is used by
the CLI in some way, so any change must be followed by the question:
Does this change in the API break the oldest supported versions of the CLI?

#### Examples of when to update

- API endpoint returns different values from before — `int` vs `string`, `tuple` vs `dict`

#### Examples of when not to update

1. Refactoring internal implementation but the endpoints and their returned values do not change
2. API supports **additional** endpoints compared to before
3. API endpoints return values change but not in a compatibility breaking way — e.g. returns a `dict` with an additional
   key. Depending on how the CLI uses the `dict` — check `dict` length vs fetch expected keys — this may be fine!

## Incrementing the package version number

Every time the version number is incremented, these parts of the codebase need to be updated:

1. `cleanlab_studio/version.py`
2. `README.md`
