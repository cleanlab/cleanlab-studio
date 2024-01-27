import os
from typing import Optional
import pandas as pd
import snowflake
from typing import Callable


def get_snowflake_datarows(
    snowflake_cursor: snowflake.connector.cursor,
    stage_name: str,
    signed_url_expiration: int = 604800,
) -> pd.DataFrame:
    """
    Returns a pandas DataFrame listing the path and presigned URLs of files stored in a Snowflake stage.
    Original implemetation here https://github.com/Labelbox/labelsnow/blob/main/labelsnow/get_snowflake_datarows.py

    Args:
        snowflake_cursor: Snowflake cursor to use for querying.
        stage_name: Name of stage to query.
        signed_url_expiration: Number of seconds until signed URLs expire. Defaults to 604800 (7 days).

    Returns:
        Pandas DataFrame with columns `external_id` and `row_data`. `external_id` is the relative path of the file in the stage. `row_data` is the presigned URL of the file.
    """
    sql_string = (
        "select relative_path as external_id, "
        "get_presigned_url(@{s_name}, relative_path, {s_expiration}) as row_data "
        "from directory(@{s_name})".format(s_name=stage_name, s_expiration=signed_url_expiration)
    )

    snowflake_cursor.execute(sql_string)
    return snowflake_cursor.fetch_pandas_all()


def get_snowflake_simple_imageset(
    snowflake_cursor: snowflake.connector.cursor,
    stage_name: str,
    root: Optional[str] = None,
    signed_url_expiration: int = 604800,
) -> pd.DataFrame:
    """
    Returns a pandas DataFrame listing the class and presigned URLs of images from an image dataset hosted on Snowflake.
    A dataset should be identified by its Snowflake stage and path from the stage to its root.

    Similar to how pytorch expects a dataset to be structured, the dataset should be structured as follows:
    https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    https://help.cleanlab.ai/guide/concepts/datasets/#simple-zip
    ```
    [root]
    ├── class_1
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── class_2
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    └── ...
    ```

    Args:
        snowflake_cursor: Snowflake cursor to use for querying.
        stage_name: Name of stage to query.
        root: Root of dataset. Defaults to None, which will treat the root of the stage as the root of the dataset.
        signed_url_expiration: Number of seconds until signed URLs expire. Defaults to 604800 (7 days).

    Returns:
        Pandas DataFrame with columns `class` and `row_data`. `class` is the class of the image. `row_data` is the presigned URL of the image.
    """
    root = os.path.normpath(root) if root is not None else ""
    sql_string = (
        "select reverse(split_part(reverse(relative_path), '/', 2)) as class, "
        "get_presigned_url(@{s_name}, relative_path, {s_expiration}) as row_data "
        "from directory(@{s_name})"
        "where relative_path like '{root}%'"
        "and relative_path not like '%/'".format(
            s_name=stage_name, s_expiration=signed_url_expiration, root=root
        )
    )

    snowflake_cursor.execute(sql_string)
    return snowflake_cursor.fetch_pandas_all()


def get_snowflake_metadata_imageset(
    snowflake_cursor: snowflake.connector.cursor,
    stage_name: str,
    metadata_path: str = "metadata.csv",
    signed_url_expiration: int = 604800,
) -> pd.DataFrame:
    """
    Returns a pandas DataFrame, listing pre-signed url of images, from an image dataset stored on snowflake and described by a metadata file. The meta data file and directory storing it should match the description from https://help.cleanlab.ai/guide/concepts/datasets/#metadata-zip.

        Args:
            snowflake_cursor: Snowflake cursor to use for querying.
            stage_name: Name of stage to query.
            metadata_path: Path to metadata file in provided stage. Defaults to "metadata.csv".
            signed_url_expiration: Number of seconds until signed URLs expire. Defaults to 604800 (7 days).

        Returns:
            Pandas DataFrame of original metadata csv, with `row_data` column added as presigned URLs for the image.
    """

    metadata_path = os.path.normpath(metadata_path)

    parent: Callable[[str], str] = lambda p: os.path.normpath(os.path.join(p, os.pardir))
    parent_of_parent = parent(parent(metadata_path))
    parent_of_parent = "" if parent_of_parent == "." else f"{parent_of_parent}/"

    snowflake_cursor.execute((f"get @{stage_name}/{metadata_path} file://."))
    df = pd.read_csv("metadata.csv")

    def get_presigned_url(filepath: str) -> str:
        snowflake_cursor.execute(
            f"select get_presigned_url(@{stage_name}, '{parent_of_parent}{filepath}', {signed_url_expiration})"
        )
        return snowflake_cursor.fetchone()[0]

    df["row_data"] = df["filepath"].apply(get_presigned_url)

    return df
