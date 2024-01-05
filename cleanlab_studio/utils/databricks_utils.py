import os
import zipfile
from datetime import datetime
from ..studio.studio import Studio


def upload_imageset_from_path(api_key, folder_path, dataset_name=None) -> str:
    """
    Uploads an imageset stored on Databricks to Cleanlab Studio.
    The imageset folder should match the layout described in the Cleanlab Studio documentations.
    https://help.cleanlab.ai/guide/concepts/datasets/#image

    Args:
        api_key: The same api_key used to initialize your Studio instance.
        folder_path: The POSIX-style absolute path to your imageset.
        dataset_name: A name you choose to use on Cleanlab studio for the imageset.

    Returns:
        ID of uploaded dataset.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    dataset_name = (
        dataset_name
        if dataset_name is not None
        else f'{folder_name}_archive_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}'
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

    studio = Studio(api_key)
    dataset_id = studio.upload_dataset(
        dataset=os.path.join(os.getcwd(), output_filename), dataset_name=dataset_name
    )
    os.remove(os.path.join(os.getcwd(), output_filename))

    return dataset_id
