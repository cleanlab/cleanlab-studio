editable:
	pip install --editable .

login:
	cleanlab login --key [api_key]

upload_csv:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv -m text --id-column tweet_id

download_combine:
	cleanlab experiment download --combine -f ./tests/resources/datasets/sample.csv --id [experiment_id]

download_csv:
	cleanlab experiment download --id [experiment_id]

check_csv:
	cleanlab dataset schema check -f ./tests/resources/datasets/sample.csv -s schema.json

upload_large:
	cleanlab dataset upload -f ./tests/resources/datasets/Tweets.csv -m text --id-column tweet_id

resume_large:
	cleanlab dataset upload -f ./tests/resources/datasets/Tweets.csv --id [dataset_id]

upload_json:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.json -m text --id-column tweet_id

resume_csv:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv --id [dataset_id]

upload_excel:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.xlsx -m text --id-column tweet_id

schema_csv:
	cleanlab dataset schema generate -f ./tests/resources/datasets/sample.csv --id-column tweet_id --modality text --name sample

schema_excel:
	cleanlab dataset schema generate -f ./tests/resources/datasets/sample.xlsx --id-column tweet_id --modality text --name sample

schema_validate:
	cleanlab dataset schema validate --schema schema.json -f ./tests/resources/datasets/sample.csv

schema_upload:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv --schema schema.json
