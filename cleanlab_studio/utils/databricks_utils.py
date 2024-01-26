from typing import Optional
import zipfile
from datetime import datetime
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, BinaryType


def dbfs_to_posix_path(dbfs_path: Path) -> Path:
    """
    Converts a DBFS path to a POSIX path.

    Args:
        dbfs_path: The DBFS path to convert.

    Returns:
        The POSIX path. Returns original path if it's not a dbfs path.
    """
    first_part = dbfs_path.parts[0]
    if first_part == "dbfs:":
        return Path("/dbfs").joinpath(dbfs_path.relative_to(first_part))
    return dbfs_path


def get_databricks_imageset_df_image_col(df: DataFrame) -> Optional[str]:
    # check for image column
    required_image_fields = [
        StructField("origin", StringType(), True),
        StructField("height", IntegerType(), True),
        StructField("width", IntegerType(), True),
        StructField("nChannels", IntegerType(), True),
        StructField("mode", IntegerType(), True),
        StructField("data", BinaryType(), True),
    ]

    struct_col_schemas = [f for f in df.schema.fields if isinstance(f.dataType, StructType)]

    for s in struct_col_schemas:
        s_fields = s.dataType.fields
        if all([f in s_fields for f in required_image_fields]):
            return s.name

    return None


def create_path_based_imageset_archive(folder_path: str, archive_name: Optional[str] = None) -> str:
    """
    Archives an imageset stored on Databricks which can then be uploaded Cleanlab Studio.
    The imageset folder should match the layout described in the Cleanlab Studio documentations.
    https://help.cleanlab.ai/guide/concepts/datasets/#image

    Args:
        folder_path: The POSIX-style path to your imageset.
        archive_name: A name you choose for the imageset archive.

    Returns:
        The relative path to the archived imageset.
    """
    folder_name = Path(folder_path).resolve(strict=False).name
    archive_name = (
        archive_name
        if archive_name is not None
        else f'{folder_name}_archive_{datetime.now().strftime("%Y-%d-%m_%H:%M:%S")}'
    )
    output_filename = f"{archive_name}.zip"

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for file in Path(folder_path).glob("**/*"):
            if file.is_file():
                # Add file to the zip file
                # The arcname argument sets the name within the zip file
                relpath = file.relative_to(Path(folder_path))
                arcname = Path(folder_name).joinpath(relpath).as_posix()
                zipf.write(file, arcname=arcname)

    return output_filename


def create_df_based_imageset_archive(df: DataFrame, archive_name: Optional[str] = None) -> str:
    """
    Archives an imageset described by a pyspark DataFrame stored on Databricks which can then be uploaded Cleanlab Studio.

    Args:
        df: PySpark DataFrame with columns `label` and `filepath`. `label` is the label of the image. `filepath` is the presigned URL of the image.
        archive_name: A name you choose for the imageset archive.

    Returns:
        The relative path to the archived imageset.
    """
    archive_name = (
        archive_name
        if archive_name is not None
        else f'upload_archive_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}'
    )
    output_filename = f"{archive_name}.zip"
    image_col = get_databricks_imageset_df_image_col(df)

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf, zipf.open(
        Path(archive_name).joinpath("metadata.csv").as_posix(), "w"
    ) as metadata_file:
        first_row = True
        for row in df.toLocalIterator():
            row = row.asDict()
            original_path = dbfs_to_posix_path(Path(row[image_col].origin))
            del row[image_col]
            path_in_zip = (
                Path(archive_name).joinpath(original_path.resolve(strict=False).name).as_posix()
            )
            row["filepath"] = path_in_zip

            # write row to a csv file called metadata.csv
            if first_row:
                metadata_file.write((",".join(map(str, row.keys())) + "\n").encode("utf-8"))
                first_row = False
            metadata_file.write((",".join(map(str, row.values())) + "\n").encode("utf-8"))
        metadata_file.close()

        for row in df.toLocalIterator():
            row = row.asDict()
            original_path = dbfs_to_posix_path(Path(row[image_col].origin))
            path_in_zip = (
                Path(archive_name).joinpath(original_path.resolve(strict=False).name).as_posix()
            )

            # add row image to the zip file
            zipf.write(original_path.as_posix(), arcname=path_in_zip)

    return output_filename
