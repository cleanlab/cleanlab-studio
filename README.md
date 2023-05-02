# cleanlab-studio [![Build Status](https://github.com/cleanlab/cleanlab-studio/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab-studio/actions?query=workflow%3ACI) [![PyPI](https://img.shields.io/pypi/v/cleanlab-studio.svg)][PyPI]

Command line and Python library interface to [Cleanlab Studio](https://cleanlab.ai/studio/). Upload datasets and download cleansets (cleaned datasets) from Cleanlab Studio in a single line of code!

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reference](#reference)

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

### Python API

You can find your API key at
```python
import cleanlab_studio

# create your Cleanlab Studio API client with your API key, found here: https://app.cleanlab.ai/account
studio = Studio(<your API key>)

# upload your dataset via a filepath, Pandas DataFrame, or PySpark DataFrame!
dataset_id: str = studio.upload_dataset(<your dataset>, <your dataset name>)

# navigate to Cleanlab Studio, create a project, and improve your labels

# download your cleanset or apply corrections to your local Pandas or PySpark dataset!
# @ANISH !!! how do we want to tell users to find their cleanset ID?
# I think we should either: Add the cleanset ID to the version history page or display it somewhere obvious on the project page
cleanset = studio.download_cleanlab_columns(<your cleanset ID>)
corrected_dataset = studio.apply_corrections(<your_dataset>, <your cleanset ID>)
```

### CLI
1. If this is your first time using the Cleanlab CLI, authenticate with `cleanlab login`. You can find your API key at https://app.cleanlab.ai/account.
2. Upload your dataset (image, text, or tabular) using `cleanlab dataset upload`.
3. Create a project in Cleanlab Studio.
4. Improve your dataset in Cleanlab Studio (e.g., correct some labels).
5. Download your cleanset with `cleanlab cleanset download`.

## Dataset Structure
Cleanlab Studio supports the following upload types:
- Text/Tabular
  - CSV
  - JSON
  - XLS/XLSX
  - Pandas DataFrame *(Python library only)*
  - PySpark DataFrame *(Python library only)*
  - more to come!
- Image
  - CSV (external media)
  - JSON (external media)
  - XLS/XLSX (external media)
  - Pandas DataFrame (external media) *(Python library only)*
  - PySpark DataFrame (external media) *(Python library only)*
  - Simple ZIP upload
  - Metadata ZIP upload

Information on dataset structuring can be found by clicking the tutorial on https://app.cleanlab.ai/upload!
## Schema

@angela TODO!

Your schema file should be formatted as follows:

```
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
    },
  },
  "version": "0.1.12"
}
```

This is the schema of a hypothetical dataset `Tweets.csv` that contains tweets, where the column `tweet_id` contains a
unique identifier for each record. Each column in the dataset is specified under `fields` with its data type and feature
type.

### Data types and Feature types

**Data type** refers to the type of the field's values: string, integer, float, or boolean.

Note that the integer type is partially *strict*, meaning floats that are equal to integers (e.g. `1.0`, `2.0`, etc)
will be accepted, but floats like `0.8` and `1.5` will not. In contrast, the float type is *lenient*, meaning integers
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

| Data type  | Feature type                                         |
|:-----------|:-----------------------------------------------------|
| string     | text, categorical, datetime, identifier, filepath    |
| integer    | categorical, datetime, identifier, numeric           |
| float      | datetime, numeric                                    |
| boolean    | boolean                                              |

The `datetime` type should be used for datetime strings, e.g. "2015-02-24 11:35:52 -0800", and Unix timestamps (which
will be integers or floats). Datetime values must be parsable
by [pandas.to_datetime()](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html).

`version` indicates the version of the Cleanlab CLI package version used to generate the schema.

[PyPI]: https://pypi.org/project/cleanlab-studio/
