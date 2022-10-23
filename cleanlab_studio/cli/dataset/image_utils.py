import os.path

from cleanlab_studio.cli.types import ImageFileExtension
from PIL import Image
from pathlib import Path

from cleanlab_studio.cli.util import get_image_file_extension


def get_image_filepath(image_filepath: str, dataset_filepath: str) -> str:
    if os.path.isabs(image_filepath):
        return image_filepath
    else:
        dataset_path = Path(dataset_filepath)
        directory_path = dataset_path.parent.absolute()
        return os.path.join(directory_path, image_filepath)


def image_file_exists(image_filepath: str, dataset_filepath: str) -> bool:
    return os.path.exists(get_image_filepath(image_filepath, dataset_filepath))


def image_file_readable(image_filepath: str, dataset_filepath: str) -> bool:
    """
    readable == image file can be opened by Pillow
    """
    try:
        image_filepath = get_image_filepath(image_filepath, dataset_filepath)
        Image.open(image_filepath)
    except IOError:
        return False
    return True
