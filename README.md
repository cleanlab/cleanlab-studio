# cleanlab-studio [![Build Status](https://github.com/cleanlab/cleanlab-studio/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab-studio/actions?query=workflow%3ACI) [![PyPI](https://img.shields.io/pypi/v/cleanlab-studio.svg)][PyPI] [![py\_versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/pypi/cleanlab-studio/)

Command line and Python library interface to [Cleanlab Studio](https://cleanlab.ai/studio/). Analyze datasets and produce *cleansets* (cleaned datasets) with Cleanlab Studio in a single line of code!

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Advanced Usage](#advanced-usage)
- [Documentation/Tutorials](https://help.cleanlab.ai/)


## Installation

You can install the Cleanlab Studio client [from PyPI][PyPI] with:

```bash
pip install cleanlab-studio
```

If you already have the client installed and wish to upgrade to the latest version, run:

```bash
pip install --upgrade cleanlab-studio
```


## Quickstart

### Python API -- run Cleanlab Studio from Python or Jupyter Notebook

```python
from cleanlab_studio import Studio

# create your Cleanlab Studio API client with your API key, found here: https://app.cleanlab.ai/account
studio = Studio(<your api key>)

# upload your dataset via a filepath, Pandas DataFrame, or PySpark DataFrame!
dataset_id: str = studio.upload_dataset(<your dataset>, <your dataset name>)

# navigate to Cleanlab Studio, create a project, and improve your labels

# download your cleanset or apply corrections to your local Pandas or PySpark dataset!
# you can find your cleanset ID by clicking on the Export Cleanset button in your project
cleanset = studio.download_cleanlab_columns(<your cleanset id>)
corrected_dataset = studio.apply_corrections(<your cleanset id>, <your dataset>)
```

The Python API offers significantly more functionality than is available in the Cleanlab Studio Web Application. Learn more via the [tutorials](https://help.cleanlab.ai/tutorials/) and [quickstart guide](https://help.cleanlab.ai/guide/quickstart/api/).

### CLI -- run Cleanlab Studio from your command line

1. If this is your first time using the Cleanlab CLI, authenticate with `cleanlab login`. You can find your API key at https://app.cleanlab.ai/account.
2. Upload your dataset (image, text, or tabular) using `cleanlab dataset upload`.
3. Create a project in Cleanlab Studio.
4. Improve your dataset in Cleanlab Studio (e.g., correct some labels).
5. Download your cleanset with `cleanlab cleanset download`.

Learn more about the Cleanlab Studio CLI from the [quickstart guide](https://help.cleanlab.ai/guide/quickstart/cli/).

## Dataset Structure

Cleanlab Studio supports the following data formats:

- Text/Tabular
  - CSV
  - JSON
  - XLS/XLSX
  - Pandas DataFrame _(Python library only)_
  - PySpark DataFrame _(Python library only)_
  - more to come!
- Image
  - CSV (external media)
  - JSON (external media)
  - XLS/XLSX (external media)
  - Pandas DataFrame (external media) _(Python library only)_
  - PySpark DataFrame (external media) _(Python library only)_
  - Simple ZIP upload
  - Metadata ZIP upload
  - more to come!

Information on how to format your dataset can be found by clicking the tutorial on https://app.cleanlab.ai/upload or in the [Datasets concept guide](https://help.cleanlab.ai/guide/concepts/datasets/). We also provide tutorials for converting certain common Python [image](https://help.cleanlab.ai/tutorials/format_image_data/) or [text](https://help.cleanlab.ai/tutorials/format_text_data/) datasets.

## Advanced Usage

### Schema

#### Python API

All schema information will be inferred by default when uploading a dataset through the Python API. We provide some options to override the inferred schema if necessary:

- To override the dataset modality, supply a `modality` kwarg to `studio.upload_dataset()`. Supported modalities include "text", "tabular", and "image"
- To override the ID column, supply an `id_column` kwarg to `studio.upload_dataset()`
- To override column types in your dataset, supply a `schema_overrides` kwarg to `studio.upload_dataset()` in the following format:

```
{
  <name_of_column_to_override>: {
    "data_type": <desired_data_type>,
    "feature_type": <desired_feature_type>,
  },
  ...
}
```

#### CLI

To specify the column types in your dataset, create a JSON file named `schema.json`. If you would like to edit an inferred schema (rather than starting from scratch) follow these steps:

1. Kick off a dataset upload using: `cleanlab dataset upload`
2. Once schema generation is complete, you'll be asked whether you'd like to use our inferred schema. Enter `n` to decline
3. You'll then be asked whether you'd like to save the inferred schema. Enter `y` to accept. Then enter the filename you'd like to save to (`schema.json` by default)
4. Edit the schema file as you wish
5. Kick off a dataset upload again using: `cleanlab dataset upload --schema_path [path to schema file]`

Your schema file should be formatted as follows:

```json
{
  "metadata": {
    "id_column": "tweet_id",
    "modality": "text",
    "name": "Tweets.csv"
  },
  "fields": {
    "tweet_id": {
      "data_type": "string",
      "feature_type": "identifier"
    },
    "sentiment": {
      "data_type": "string",
      "feature_type": "categorical"
    },
    "sentiment_confidence": {
      "data_type": "float",
      "feature_type": "numeric"
    },
    "retweet_count": {
      "data_type": "integer",
      "feature_type": "numeric"
    },
    "text": {
      "data_type": "string",
      "feature_type": "text"
    },
    "tweet_created": {
      "data_type": "boolean",
      "feature_type": "boolean"
    },
    "tweet_created": {
      "data_type": "string",
      "feature_type": "datetime"
    }
  },
  "version": "0.1.12"
}
```

This is the schema of a hypothetical dataset `Tweets.csv` that contains tweets, where the column `tweet_id` contains a
unique identifier for each record. Each column in the dataset is specified under `fields` with its data type and feature
type.

#### Data types and Feature types

**Data type** refers to the type of the field's values: string, integer, float, or boolean.

Note that the integer type is partially _strict_, meaning floats that are equal to integers (e.g. `1.0`, `2.0`, etc)
will be accepted, but floats like `0.8` and `1.5` will not. In contrast, the float type is _lenient_, meaning integers
are accepted. Users should select the float type if the field may include float values. Note too that integers can have
categorical and identifier feature types, whereas floats cannot.

For booleans, the list of accepted values are: true/false, t/f, yes/no, 1/0, 1.0/0.0.

**Feature type** refers to the secondary type of the field, relating to how it is used in a machine learning model, such
as whether it is:

- a categorical value
- a numeric value
- a datetime value
- a boolean value
- text
- an identifier — a string / integer that identifies some entity
- a filepath value (only valid for image datasets)

Some feature types can only correspond to specific data types. The list of possible feature types for each data type is
shown below

| Data type | Feature type                                      |
| :-------- | :------------------------------------------------ |
| string    | text, categorical, datetime, identifier, filepath |
| integer   | categorical, datetime, identifier, numeric        |
| float     | datetime, numeric                                 |
| boolean   | boolean                                           |

The `datetime` type should be used for datetime strings, e.g. "2015-02-24 11:35:52 -0800", and Unix timestamps (which
will be integers or floats). Datetime values must be parseable
by [polars.from_epoch](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.from_epoch.html) for integer/floats or [polars.Expr.str.strptime](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.str.strptime.html) for strings.

`version` indicates the version of the Cleanlab CLI package version used to generate the schema.

## Other Resources

- [Documentation](https://help.cleanlab.ai/) -- tutorials on how to use Cleanlab Studio and guides to learn key concepts
- [Blog](https://cleanlab.ai/examples/) -- end-to-end applications, feature announcements, how-it-works explanations, benchmarks
- [Slack Community](https://cleanlab.ai/slack/) -- ask questions, request features, discuss Data-Centric AI with others
- Need professional help or want demo? Reach out via email: team@cleanlab.ai

<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/studio.png" width=80% height=80%>
</p>

[PyPI]: https://pypi.org/project/cleanlab-studio/
