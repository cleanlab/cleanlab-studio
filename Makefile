editable:
	pip install --editable .

auth:
	cleanlab auth --key [key]

upload_csv:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv -m text --id_column tweet_id

upload_json:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.json -m text --id_column tweet_id

resume_csv:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv --id [dataset_id]

upload_excel:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.xlsx -m text --id_column tweet_id

schema_csv:
	cleanlab dataset schema generate -f ./tests/resources/datasets/sample.csv --id_column tweet_id --modality text --name sample

schema_excel:
	cleanlab dataset schema generate -f ./tests/resources/datasets/sample.xlsx --id_column tweet_id --modality text --name sample

schema_validate:
	cleanlab dataset schema validate --schema schema.json -d ./tests/resources/datasets/sample.csv

schema_upload:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv --schema schema.json
