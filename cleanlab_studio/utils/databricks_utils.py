import os
import zipfile
from datetime import datetime
from pathlib import Path

from cleanlab_studio.internal.util import get_databricks_imageset_df_image_col


def dbfs_to_posix_path(dbfs_path: str) -> str:
    """
    Converts a DBFS path to a POSIX path.

    Args:
        dbfs_path: The DBFS path to convert.

    Returns:
        The POSIX path.
    """
    return f"/dbfs{dbfs_path[5:]}"


def create_path_based_imageset_archive(folder_path, archive_name=None) -> str:
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
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Create a full path
                full_path = Path(root).joinpath(file).as_posix()
                # Add file to the zip file
                # The arcname argument sets the name within the zip file
                relpath = Path(full_path).relative_to(Path(folder_path))
                arcname = Path(folder_name).joinpath(relpath).as_posix()
                zipf.write(full_path, arcname=arcname)

    return output_filename


def create_df_based_imageset_archive(df, archive_name=None) -> str:
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
        f"{archive_name}/metadata.csv", "w"
    ) as metadata_file:
        first_row = True
        for row in df.toLocalIterator():
            row = row.asDict()
            original_path = dbfs_to_posix_path(row[image_col].origin)
            path_in_zip = Path(archive_name).joinpath(Path(original_path).name).as_posix()
            row[image_col] = path_in_zip

            # write row to a csv file called metadata.csv
            if first_row:
                metadata_file.write((",".join(map(str, row.keys())) + "\n").encode("utf-8"))
                first_row = False
            metadata_file.write((",".join(map(str, row.values())) + "\n").encode("utf-8"))
        metadata_file.close()

        for row in df.toLocalIterator():
            row = row.asDict()
            original_path = dbfs_to_posix_path(row[image_col].origin)
            path_in_zip = Path(archive_name).joinpath(Path(original_path).name).as_posix()

            # add row image to the zip file
            zipf.write(original_path, arcname=path_in_zip)

    return output_filename
