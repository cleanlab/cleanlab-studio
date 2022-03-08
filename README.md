# cleanlab-cli
Command line interface for all things Cleanlab


### Schema

To specify the column types in your dataset, create a JSON file named `schema.json` with the following format:
```
{
  "fields": [
    {
      "name": "tweet_id",
      "type": "string"
    },
    {
      "name": "probability",
      "type": "float"
    },
    {
      "name": "retweet_count",
      "type": "integer"
    },
    {
      "name": "text",
      "type": "string"
    },
    {
      "name": "retweeted",
      "type": "boolean"
    },
    {
      "name": "date_created",
      "type": "datetime"
    }
  ],
  "version": 1.0
}
```
Above is the schema of a hypothetical dataset containing tweets. 
Column datatypes are specified under `fields` as an array of objects with the keys `name` and `type`.
The list of accepted types are: `string`, `integer`, `float`, `boolean`, and `datetime`.

The `datetime` type should be used for datetime strings, e.g. "2015-02-24 11:35:52 -0800". 
For Unix timestamps, use the `integer` type.

`version` indicates the current Cleanlab schema version at time of schema creation. 
The current Cleanlab schema version is `1.0`.

The CLI can generate a proposed schema for you: `cleanlab dataset schema --file <filepath> --output <output_filepath>`.
By default, `cleanlab dataset schema --file <filepath>` outputs a schema to `schema.json`.

### For developers
1. Set up and activate your virtual environment
2. When in active development, use `pip install --editable .` from the home directory, 
so that changes to the code are reflected automatically in the CLI.
