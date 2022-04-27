from click.testing import CliRunner
from cleanlab_cli.dataset.schema import generate_schema_command, validate_schema_command
from cleanlab_cli.dataset.util import read_file_as_df
import os
from os.path import dirname, abspath
import logging

logger = logging.getLogger(__name__)

sample_csv = os.path.join(abspath(dirname(__file__)), "resources/datasets/sample.csv")
sample_schema = os.path.join(abspath(dirname(__file__)), "resources/schemas/sample_schema.json")


def assert_success_else_error_output(test_name, result):
    if result.exit_code != 0:
        raise AssertionError(
            f"{test_name} failed with exit code {result.exit_code} and output: {result.output}"
        )


def test_schema():
    df = read_file_as_df(sample_csv)
    runner = CliRunner()
    with runner.isolated_filesystem():
        filename = "sample"

        df.to_csv(filename + ".csv")
        df.to_excel(filename + ".xlsx", index=False)
        # df.to_json(filename + ".json", orient="table", index=False)

        for ext in [".csv", ".xlsx"]:
            result = runner.invoke(
                generate_schema_command,
                [
                    "-f",
                    filename + ext,
                    "--id-column",
                    "tweet_id",
                    "--modality",
                    "text",
                    "--output",
                    "schema.json",
                ],
            )
            assert_success_else_error_output("Schema generation", result)
            result = runner.invoke(
                validate_schema_command, ["--schema", sample_schema, "--dataset", filename + ext]
            )
            assert_success_else_error_output("Schema validation", result)


if __name__ == "__main__":
    test_schema()
