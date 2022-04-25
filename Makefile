editable:
	pip install --editable .

login:
	cleanlab login --key 741361c80b114518b4ff4a23045e2c18

upload_csv:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv -m text --id-column tweet_id

upload_large:
	cleanlab dataset upload -f ./tests/resources/datasets/Tweets.csv -m text --id-column tweet_id

resume_large:
	cleanlab dataset upload -f ./tests/resources/datasets/Tweets.csv --id fef88b5c46824c628b80130194a9fc45

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
	cleanlab dataset schema validate --schema schema.json -d ./tests/resources/datasets/sample.csv

schema_upload:
	cleanlab dataset upload -f ./tests/resources/datasets/sample.csv --schema schema.json
