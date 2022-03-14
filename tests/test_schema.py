from click.testing import CliRunner
from dataset.schema import generate_schema_command, validate_schema_command
from dataset.util import read_file_as_df

sample_csv = "resources/datasets/sample.csv"


def test_generate_schema():
    df = read_file_as_df(sample_csv)
    runner = CliRunner()
    with runner.isolated_filesystem():
        filename = "sample"

        df.to_csv(filename + ".csv")
        df.to_excel(filename + ".xlsx", index=False)
        df.to_json(filename + ".json", orient="table", index=False)

        for ext in [".csv", ".xlsx"]:
            result = runner.invoke(
                generate_schema_command,
                ["-f", filename + ext, "--id_col", "tweet_id", "--modality", "text"],
            )
            try:
                assert result.exit_code == 0
            except AssertionError:
                raise AssertionError(
                    f"Generate schema for {ext} failed with exit code: {result.exit_code},"
                    f" exception: {result.exception}"
                )

            result = runner.invoke(
                validate_schema_command, ["-f", filename + ext, "--schema", "schema.json"]
            )
            try:
                assert result.exit_code == 0
            except AssertionError:
                raise AssertionError(
                    f"Validate schema for {ext} failed with exit code: {result.exit_code},"
                    f" exception: {result.exception}"
                )


if __name__ == "__main__":
    test_generate_schema()
