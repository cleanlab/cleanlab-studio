# cleanlab-cli
Command line interface for all things Cleanlab Studio.

Currently the main thing this supports is getting your data into CLeanlab Studio and downloading your results. In the future, this will also support computation.


### Schema

To specify the column types in your dataset, create a JSON file named `schema.json` with the following format:
```
{
  "fields": {
    "tweet_id": "string",
    "probability": "float",
    "retweet_count": "integer",
    "text": "string",
    "retweeted": "boolean",
    "date_created": "datetime"
  },
  "version": 1.0
}
```
Above is the schema of a hypothetical dataset containing tweets. 
Column datatypes are specified under `fields` as an object with column names as keys and data types as values.
The list of accepted types are: `string`, `integer`, `float`, `boolean`, and `datetime`.

The `datetime` type should be used for datetime strings, e.g. "2015-02-24 11:35:52 -0800", and Unix timestamps.

`version` indicates the current Cleanlab schema version at time of schema creation. 
The current Cleanlab schema version is `1.0`.

The CLI can generate a proposed schema for you: 

`cleanlab dataset schema --file <filepath>`.

By default, this outputs a schema to `schema.json`. You may further configure this with `--output <filepath>`.

### For developers
1. Set up and activate your virtual environment
2. When in active development, use `pip install --editable .` from the home directory, 
so that changes to the code are reflected automatically in the CLI.
