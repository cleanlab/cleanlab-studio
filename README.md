# cleanlab-studio [![Build Status](https://github.com/cleanlab/cleanlab-studio/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab-studio/actions?query=workflow%3ACI) [![PyPI](https://img.shields.io/pypi/v/cleanlab-studio.svg)][PyPI]

Command line interface for all things Cleanlab Studio.

This currently supports generating <a href="#schema">dataset schema</a>, uploading <a href="#dataset-format">
datasets</a> into Cleanlab Studio, and downloading cleansets from Cleanlab Studio.

## Installation

You can install the Cleanlab Studio CLI [from PyPI][PyPI] with:

```bash
pip install cleanlab-studio
```

If you already have the CLI installed and wish to upgrade to the latest version, run:

```bash
pip install --upgrade cleanlab-studio
```

## Workflow

Uploading datasets to Cleanlab Studio is a two-step process.

1. We generate a schema describing the dataset and its <a href="#data-types-and-feature-types">data and feature
   types</a>, which is verified by the user.
2. Based on this schema, the dataset is parsed and uploaded to Cleanlab Studio.

### Upload a dataset

To upload a dataset without
first [generating a schema](https://github.com/cleanlab/cleanlab-studio/#generate-dataset-schema) (i.e. Cleanlab will
suggest one for you):

`cleanlab dataset upload -f [dataset filepath]`

**Note:** For image datasets, `[dataset filepath]` refers to the path of the labels file. <a href="#dataset-format">Learn more about the labels file.</a>

You will be asked to `"Specify your dataset modality (text, tabular, image):"`.

* Enter `text` to only find label errors based on a single column of text in your dataset.
* Enter `tabular` to find data and label issues based on any subset of the column features.
* Enter `image` to find label errors in your image dataset.

To upload a dataset with a schema:

`cleanlab dataset upload -f [dataset filepath] -s [schema filepath]`

To **resume uploading** a dataset whose upload was interrupted:

`cleanlab dataset upload -f [dataset filepath] --id [dataset ID]`

A dataset ID is generated and printed to the terminal the first time the dataset is uploaded. It can also be accessed from the Datasets section of the Cleanlab Studio dashboard by visiting https://app.cleanlab.ai/ and selecting 'Resume' for the relevant dataset.

#### Generate dataset schema

To generate a dataset schema (prior to uploading your dataset):

`cleanlab dataset schema generate -f [dataset filepath]`

* For `Modality (text, tabular, image): ` enter one of the following:
  * `text` - to only find label errors based on a single column of text in your dataset.
  * `tabular` - to find data and label issues based on any subset of the column features.
  * `image` - to find label errors in your image dataset.

* For `Id column: `, please enter the string name of of the column in your dataset that contains the id of each row.

* For `filepath column: ` (only for image modality), please enter the string name of of the column in your dataset that contains the path to each image.


To validate an existing schema, i.e. check that it is complete, well-formatted, and
has <a href="#data-types-and-feature-types">data types with sensible feature types</a>:

`cleanlab dataset schema validate -s [schema filepath]`

You may then wish to inspect the generated schema to check that the fields and metadata are correct.

### Download clean labels

To download clean labels (i.e. labels that have been fixed through the Cleanlab Studio interface):

`cleanlab cleanset download --id [cleanset ID]`

To download clean labels and combine them with your local dataset:

`cleanlab cleanset download --id [cleanset ID] -f [dataset filepath]`

## Commands

**`cleanlab login` authenticates you**

Authenticates you when uploading datasets to Cleanlab Studio. Pass in your API key using `--key [API key]`. Your API key
can be accessed at [https://app.cleanlab.ai/upload](https://app.cleanlab.ai/upload).

**`cleanlab dataset schema generate` generates dataset schemas**

Generates a schema based on your dataset. Specify your target dataset with `--filepath [dataset filepath]`. You will be
prompted to save the generated schema JSON and to specify a save location. This can be specified
using `--output [output filepath]`.

**`cleanlab dataset schema validate` validates a schema JSON file**

Validates a schema JSON file, checking that a schema is complete, well-formatted, and
has <a href="#data_types_and_feature_types">data types with sensible feature types</a>. Specify your target schema
with `--schema [schema filepath]`.

You may also validate an existing schema with respect to a dataset (`-d [dataset filepath]`), i.e. all previously
mentioned checks and the additional check that all fields in the schema are present in the dataset.

**`cleanlab dataset upload` uploads your dataset**

Uploads your dataset to Cleanlab Studio. Specify your target dataset with `--filepath [dataset filepath]`. You will be
prompted for further details about the dataset's modality and ID column. These may be supplied to the command
with `--modality [modality]`, `--id-column [name of ID column]`, and you may also specify a custom dataset name
with`--name [custom dataset name]`.

After uploading your dataset, you will be prompted to save the list of dataset issues (if any) encountered during the
upload process. These issues include missing IDs, duplicate IDs, missing values, and values whose types do not match the
schema. You may specify the save location with `--output [output filepath]`.

**`cleanlab cleanset download` downloads Cleanlab columns from your cleanset**

Cleansets are initialized through the Cleanlab Studio interface. In a cleanset, users can inspect their dataset and
verify their labels. Clean labels are the labels after this set of manual fixes have been applied.

This command downloads the clean labels and saves them locally as a .csv, .xls/.xlsx, or .json, with columns `id`
and `clean_label`. Include the `--filepath [dataset filepath]` to combine the clean labels with the original dataset as
a new column `clean_label`, which will be outputted to `--output [output filepath]`. Include the `--all` flag to
include **all** Cleanlab columns, i.e. issue, label quality, suggested label, clean label, instead of only the clean
label column.

## Dataset format

Cleanlab currently only supports text, tabular, and image dataset modalities.
(If your dataset contains both text and tabular data, treat it as tabular.)

The accepted dataset file types are: `.csv`, `.json`, and `.xls/.xlsx`.

Check below for instructions of formatting different dataset modalities.

#### Tabular dataset format
- dataset must have an **ID column** (`flower_id` in the example below) - a column containing identifiers that uniquely identify each row.
- dataset must have a **label column** (`species` in the example below) for the prediction task.
- Apart from the reserved column name: `clean_label`, you are free to name the columns in your dataset in any way you want.

###### .csv, .xls/.xlsx

| flower_id | width | length | color | species |
|:----------|:------|--------|-------|---------|
| flower_01 | 4     | 3      | red   | rose    |
| flower_02 | 7     | 2      | white | lily    |

###### .json

```json
{
  "rows": [
    {
      "flower_id": "flower_01",
      "width": 4,
      "length": 3,
      "color": "red",
      "species": "rose"
    },
    {
      "flower_id": "flower_02",
      "width": 7,
      "length": 2,
      "color": "white",
      "species": "lily"
    }
  ]
}
```
<br />

#### Text dataset format

- dataset must have an **ID column** (`review_id` in the example below) - a column containing identifiers that uniquely identify each row.
- dataset must have a **text column** (`review` in the example below) that serves as the input set for the prediction task.
- dataset must have a **label column** (`sentiment` in the example below) that serves as the output for the prediction task.
- Apart from the reserved column name: `clean_label`, you are free to name the columns in your dataset in any way you want.

###### .csv, .xls/.xlsx

| review_id | review | sentiment |
|:----------|:-------|-----------|
| review_1  | The sales rep was fantastic!     | positive  |
| review_2  | He was a bit wishy-washy.     | negative  |

###### .json

```json
{
  "rows": [
    {
      "review_id": "review_1",
      "review": "The sales rep was fantastic!",
      "label": "positive"
    },
    {
      "review_id": "review_2",
      "review": "He was a bit wishy-washy.",
      "label": "negative"
    }
  ]
}
```
<br />

#### Image dataset format
- Image Datasets have 2 components:
  - **Set of images.**
  - **Labels file** - A mapping from each image to a label. This mapping can be supplied either in a .csv, .xls/.xlsx, or json format.

###### Labels file format
- must have an **ID column** (`vizzy_id` in the example below) - a column containing identifiers that uniquely identify each row.
- must have a **filepath column** (`vizzy_path` in the example below) that contains path to the image file.
- must have a **label column** (`label` in the example below) that contains the label for the corresponding image file.
- may have any number of metadata columns. Apart from the reserved column name: `clean_label`, you are free to name these columns any way you want.

###### Dataset format

###### .csv, .xls/.xlsx

| vizzy_id | vizzy_path | label |
|:----------|:-------|-----------|
| 1  | Dataset/scruppy.jpeg    | cat  |
| 2  | Dataset/tuffy/fluffy.png    | cat  |
| 3  | oreo.jpeg    | dog  |
| 4  | Dataset/mocha/mocha.jpeg    | dog  |

###### .json

```json
{
  "rows": [
    {
      "vizzy_id": "1",
      "vizzy_path": "Dataset/scruppy.jpeg",
      "label": "cat"
    },
    {
      "vizzy_id": "2",
      "vizzy_path": "Dataset/tuffy/fluffy.png",
      "label": "cat"
    },
    {
      "vizzy_id": "3",
      "vizzy_path": "oreo.jpeg",
      "label": "dog"
    },
    {
      "vizzy_id": "4",
      "vizzy_path": "Dataset/mocha/mocha.jpeg",
      "label": "dog"
    }
  ]
} 
```
<br />

## Schema

To specify the column types in your dataset, create a JSON file named `schema.json`. We recommend
using `cleanlab dataset schema generate` to generate an initial schema and editing from there.

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

`version` indicates the version of the Cleanlab CLI package version used to generate the schema. The current Cleanlab
schema version is `0.1.15`.

[PyPI]: https://pypi.org/project/cleanlab-studio/
