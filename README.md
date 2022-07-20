# cleanlab-cli

Command line interface for all things Cleanlab Studio.

This currently supports generating <a href="#schema">dataset schema</a>, uploading <a href="#dataset-format">
datasets</a> into Cleanlab Studio, and downloading clean labels from Cleanlab Studio.

## Workflow

Uploading datasets to Cleanlab Studio is a two-step process.

1. We generate a schema describing the dataset and its <a href="#data-types-and-feature-types">data and feature
   types</a>, which is verified by the user.
2. Based on this schema, the dataset is parsed and uploaded to Cleanlab Studio.

### Generate dataset schema

To generate a dataset schema:

`cleanlab dataset schema generate -f [dataset filepath]`

To validate an existing schema, i.e. check that it is complete, well-formatted, and
has <a href="#data-types-and-feature-types">data types with sensible feature types</a>:

`cleanlab dataset schema validate -s [schema filepath]`

You may then wish to inspect the generated schema to check that the fields and metadata are correct.

### Upload a dataset

To upload a dataset without a schema (i.e. Cleanlab will suggest one for you):

`cleanlab dataset upload -f [dataset filepath]`

To upload a dataset with a schema:

`cleanlab dataset upload -f [dataset filepath] -s [schema filepath]`

To **resume uploading** a dataset whose upload was interrupted:

`cleanlab dataset upload -f [dataset filepath] --id [dataset ID]`

A dataset ID is generated and printed to the terminal the first time the dataset is uploaded. It can also be accessed by
visiting https://app.cleanlab.ai/datasets and selecting 'Resume' for the relevant dataset.

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

Cleanlab currently only supports text and tabular dataset modalities.
(If your dataset contains both text and tabular data, treat it as tabular.)
The accepted dataset file types are: `.csv`, `.json`, and `.xls/.xlsx`.

Below are some examples of how to format your dataset depending on modality and file type.

Every dataset must have an **ID column** (i.e. a column containing identifiers that uniquely identify each row) and a
**label column** (for the prediction task).

Apart from the reserved column name: `clean_label`, You are free to name the columns in your dataset in any way you
want.

<details>
<summary>Tabular</summary>
<br />
<details>
<summary>.csv, .xls/.xlsx</summary>

| flower_id | width | length | color | species |
|:----------|:------|--------|-------|---------|
| flower_01 | 4     | 3      | red   | rose    |
| flower_02 | 7     | 2      | white | lily    |

</details>
<details>
<summary>.json</summary>

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

</details>
</details>

<details>
<summary>Text</summary>
<br />
<details>
<summary>.csv, .xls/.xlsx</summary>

| review_id | review | sentiment |
|:----------|:-------|-----------|
| review_1  | The sales rep was fantastic!     | positive  |
| review_2  | He was a bit wishy-washy.     | negative  |

</details>

<details>
<summary>.json</summary>

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

</details>
</details>

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
  "version": "0.1.0"
}
```

This is the schema of a hypothetical dataset `Tweets.csv` that contains tweets, where the column `tweet_id` contains a
unique identifier for each record. Each column in the dataset is specified under `fields` with its data type and feature
type.

### Data types and Feature types

**Data type** refers to the type of the field's values: string, integer, float, or boolean.

Note that the integer type is *strict*, meaning floats will be rejected. In contrast, the float type is *lenient*,
meaning integers are accepted. Users should select the float type if the field may include float values. Note too that
integers can have categorical and identifier feature types, whereas floats cannot.

For booleans, the list of accepted values are: true/false, t/f, yes/no, and 1/0.

**Feature type** refers to the secondary type of the field, relating to how it is used in a machine learning model, such
as whether it is:

- a categorical value
- a numeric value
- a datetime value
- a boolean value
- text
- an identifier — a string / integer that identifies some entity

Some feature types can only correspond to specific data types. The list of possible feature types for each data type is
shown below

| Data type  | Feature type                               |
|:-----------|:-------------------------------------------|
| string     | text, categorical, datetime, identifier    |
| integer    | categorical, datetime, identifier, numeric |
| float      | datetime, numeric                          |
| boolean    | boolean                                    |

The `datetime` type should be used for datetime strings, e.g. "2015-02-24 11:35:52 -0800", and Unix timestamps (which
will be integers or floats). Datetime values must be parsable
by [pandas.to_datetime()](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html).

`version` indicates the version of the Cleanlab CLI package version used to generate the schema. The current Cleanlab
schema version is `0.1.7`.
