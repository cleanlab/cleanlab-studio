# cleanlab-studio [![Build Status](https://github.com/cleanlab/cleanlab-studio/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab-studio/actions?query=workflow%3ACI) [![PyPI](https://img.shields.io/pypi/v/cleanlab-studio.svg)][PyPI]

Command line interface for all things [Cleanlab Studio](https://cleanlab.ai/studio/). Upload datasets and download cleansets (cleaned datasets) from Cleanlab Studio in a single line of code!

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reference](#reference)

## Installation

You can install the Cleanlab Studio CLI [from PyPI][PyPI] with:

```bash
pip install cleanlab-studio
```

If you already have the CLI installed and wish to upgrade to the latest version, run:

```bash
pip install --upgrade cleanlab-studio
```

## Quickstart

1. If this is your first time using the Cleanlab CLI, authenticate with `cleanlab login`. You can find your API key at <https://app.cleanlab.ai/upload>, by clicking on "Upload via Command Line".

2. Upload your dataset ([image](#upload-an-image-dataset), [text](#upload-a-text-dataset), or [tabular](#upload-a-tabular-dataset)) using `cleanlab dataset upload`.

3. Create a project in Cleanlab Studio.

4. Improve your dataset in Cleanlab Studio (e.g., correct some labels).

5. [Download your cleanset](#download-cleanset) with `cleanlab cleanset download`.

### Upload a dataset

#### Upload an image dataset

##### Simple upload

If your dataset is organized in a particular way, you can upload it using the simple upload flow. In the simple organization, a dataset consists of folders for each class, with images in the corresponding folder. For example:

```
- animals
  - dog
    - scruffy.png
    - spot.jpg
  - cat
    - whiskers.png
    - yoda.jpg
  - snake
    - basalisk.png
    - medusa.jpg
```

A dataset formatted in this way can be uploaded with:

```bash
cleanlab dataset upload -f [dataset directory]
```

##### With any organization, with metadata

More generally, an image dataset consists of a collection of image files (organized in any way, in any folder hierarchy and with any file names), along with a metadata file specifying paths and labels (and optionally, other metadata). An image dataset might be organized like this:

```
- dogs
  - scruffy.png
  - spot.jpg
- cats
  - whiskers.png
- fred.png
- labels.csv
```

An example `labels.csv` looks like this:

```
id,path,label
1,dogs/scruffy.png,dog
2,dogs/spot.jpg,dog
3,cats/whiskers.png,cat
4,fred.png,human
```

The metadata file, `labels.csv`, must contain at least three columns:

- an ID column (with unique identifiers for each datapoint)
- a path column (with relative paths to image files)
- a label column (with categorical labels)

The metadata file is also allowed to have extra columns with various types of metadata.

Image datasets can be uploaded with:

```bash
cleanlab dataset upload --modality image -f [metadata file]
```

Follow the prompts to specify the ID column and path column.

If you have a dataset with metadata columns where this package isn't able to correctly infer the data/feature types, see the [reference](#reference) on dataset schemas.

#### Upload a text dataset

A text dataset contains a single predictve feature (text), along with labels. A text dataset should have a minimum of three columns:

- an ID column (with unique identifiers for each datapoint)
- a text column (containing text)
- a label column (with categorical labels)

The dataset is allowed to have extra columns. This package supports `.csv`, `.json`, and `.xls/.xlsx` datasets.

Text datasets can be uploaded with:

```bash
cleanlab dataset upload --modality text -f [dataset]
```

If you have a dataset with columns where this package isn't able to correctly infer the data/feature types, see the [reference](#reference) on dataset schemas.

#### Upload a tabular dataset

A tabular dataset has a number of predictive features, along with labels. A tabular dataset should have at least:

- an ID column (with unique identifiers for each datapoint)
- a label column (with categorical labels)

The dataset can have as many feature columns as you would like. This package supports `.csv`, `.json`, and `.xls/.xlsx` datasets.

Tabular datasets can be uploaded with:

```bash
cleanlab dataset upload --modality tabular -f [dataset]
```

If you have a dataset with columns where this package isn't able to correctly infer the data/feature types, see the [reference](#reference) on dataset schemas.

### Download Cleanset

To download clean labels (i.e., labels that have been fixed through the Cleanlab Studio interface), first find the ID of the cleanset. You can find this by clicking the "Export Cleanset" button in the top right of a project page.

```bash
cleanlab cleanset download --id [cleanset ID]
```

The above command only downloads corrected labels. You can also download corrected labels and combine them with your local dataset in a single command:

```bash
cleanlab cleanset download --id [cleanset ID] -f [dataset filepath]
```

Include the `--all` flag to include **all** Cleanlab columns, i.e. issue, label quality, suggested label, clean label, instead of only the clean label column.

## Reference

### Workflow

Uploading datasets to Cleanlab Studio is a two-step process.

1. Generate a schema describing the dataset and its [data and feature types](#data-types-and-feature-types)
2. Based on the schema, parse and upload the dataset to Cleanlab Studio.

#### Upload a dataset

To upload a dataset without
first [generating a schema](#generate-dataset-schema) (i.e. Cleanlab will
suggest one for you):

`cleanlab dataset upload -f [dataset filepath]`

To upload a dataset with a schema:

`cleanlab dataset upload -f [dataset filepath] -s [schema filepath]`

To **resume uploading** a dataset whose upload was interrupted:

`cleanlab dataset upload -f [dataset filepath] --id [dataset ID]`

A dataset ID is generated and printed to the terminal the first time the dataset is uploaded. It can also be accessed from the Datasets section of the Cleanlab Studio dashboard by visiting <https://app.cleanlab.ai/> and selecting "Resume" for the relevant dataset.

#### Generate dataset schema

To generate a dataset schema (prior to uploading your dataset):

`cleanlab dataset schema generate -f [dataset filepath]`

* For `Id column: `, enter the name of the column in your dataset that contains the unique identifier for each row.

Make sure to inspect the schema. If any [data/feature types](#data-types-and-feature-types) are not inferred correctly, you can edit the schema manually.

You can validate a schema with `cleanlab dataset schema validate`. You can also validate a schema with respect to a dataset by specifying the `-d [dataset filepath]` option.

### Dataset format

Cleanlab currently only supports text, tabular, and image dataset modalities.
(If your data contains both text and numeric/categorical columns, treat it as tabular.)

The accepted dataset file types are: `.csv`, `.json`, and `.xls/.xlsx`.

Each entry (i.e. row) should correspond to a different example in the dataset.

#### Tabular dataset format

- dataset must have an **ID column** (`flower_id` in the example below) - a column containing identifiers that uniquely identify each row.
- dataset must have a **label column** (`species` in the example below) which you either want to train models to predict or simply find erroneous values in.
- Apart from the reserved column name: `clean_label`, you are free to name the columns in your dataset in any way you want. There can be some subset of the columns used as features to predict the label, based upon which Cleanlab Studio identifies label issues, and other columns with extra metadata, that will be ignored when modeling the labels.

###### .csv, .xls/.xlsx

| flower_id | width | length | color | species |
|:----------|:------|--------|-------|---------|
| flower_01 | 4     | 3      | red   | rose    |
| flower_02 | 7     | 2      | white | lily    |

###### .json

```json
[
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
```
<br />

#### Text dataset format

- dataset must have an **ID column** (`review_id` in the example below) - a column containing identifiers that uniquely identify each row.
- dataset must have a **text column** (`review` in the example below) that serves as the sole predictive feature for modeling the label and identifying label issues.
- dataset must have a **label column** (`sentiment` in the example below) which you either want to train models to predict or simply find erroneous values in.
- Apart from the reserved column name: `clean_label`, you are free to name the columns in your dataset in any way you want.

###### .csv, .xls/.xlsx

| review_id | review | sentiment |
|:----------|:-------|-----------|
| review_1  | The sales rep was fantastic!     | positive  |
| review_2  | He was a bit wishy-washy.     | negative  |

###### .json

```json
[
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
```
<br />

#### Image dataset format
- Image datasets have two components:
  - **Collection of image files.**
  - **Labels file** - A mapping from each image filepath to a class label. This mapping can be supplied either in a .csv, .xls/.xlsx, or .json format.

###### Labels file format
- must have an **ID column** (`vizzy_id` in the example below) - a column containing identifiers that uniquely identify each row.
- must have a **filepath column** (`vizzy_path` in the example below) that contains relative path to the image file.
- must have a **label column** (`label` in the example below) that contains the label for the corresponding image file.
- may have any number of extra metadata columns that will not be used to model labels and identify label issues. Apart from the reserved column name: `clean_label`, you are free to name these columns any way you want.

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
[
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

`version` indicates the version of the Cleanlab CLI package version used to generate the schema.

[PyPI]: https://pypi.org/project/cleanlab-studio/
