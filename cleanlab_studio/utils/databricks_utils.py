import os
import zipfile
from tqdm import tqdm
from datetime import datetime
import csv


def create_path_based_imageset_archive(folder_path, dataset_name=None) -> str:
    """
    Archives an imageset stored on Databricks which can then be uploaded Cleanlab Studio.
    The imageset folder should match the layout described in the Cleanlab Studio documentations.
    https://help.cleanlab.ai/guide/concepts/datasets/#image

    Args:
        folder_path: The POSIX-style path to your imageset.
        dataset_name: A name you choose to use on Cleanlab studio for the imageset.

    Returns:
        The path to the archived imageset.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    dataset_name = (
        dataset_name
        if dataset_name is not None
        else f'{folder_name}_archive_{datetime.now().strftime("%Y-%d-%m_%H:%M:%S")}'
    )
    output_filename = f"{dataset_name}.zip"

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Create a full path
                full_path = os.path.join(root, file)
                # Add file to the zip file
                # The arcname argument sets the name within the zip file
                relpath = os.path.relpath(full_path, folder_path)
                arcname = os.path.join(folder_name, relpath)
                zipf.write(full_path, arcname=arcname)

    return output_filename


def create_df_based_archive(df, root="", dataset_name=None) -> str:
    """
    Archives an imageset described by a pyspark DataFrame stored on Databricks which can then be uploaded Cleanlab Studio.

    Args:
        df: PySpark DataFrame with columns `label` and `filepath`. `label` is the label of the image. `filepath` is the presigned URL of the image.
        root: The POSIX-style path to the root of the imageset. If not provided, the paths in `filepath` are assumed to be absolute.
        dataset_name: A name you choose to use on Cleanlab studio for the imageset.

    Returns:
        The path to the archived imageset.
    """
    dataset_name = (
        dataset_name
        if dataset_name is not None
        else f'upload_archive_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}'
    )
    output_filename = f"{dataset_name}.zip"

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf, zipf.open(
        f"{dataset_name}/metadata.csv", "w"
    ) as metadata_file:
        first_row = True
        for row in df.toLocalIterator():
            row = row.asDict()
            original_path = row["filepath"]
            file_name = os.path.basename(original_path)
            row["filepath"] = f"{dataset_name}/{file_name}"

            # write row to a csv file called metadata.csv
            if first_row:
                metadata_file.write((",".join(map(str, row.keys())) + "\n").encode("utf-8"))
                first_row = False
            metadata_file.write((",".join(map(str, row.values())) + "\n").encode("utf-8"))
        metadata_file.close()

        for row in df.toLocalIterator():
            row = row.asDict()
            original_path = row["filepath"]
            file_name = os.path.basename(original_path)
            row["filepath"] = f"{dataset_name}/{file_name}"

            # add row image to the zip file
            zipf.write(os.path.join(root, original_path), arcname=row["filepath"])

    return output_filename
