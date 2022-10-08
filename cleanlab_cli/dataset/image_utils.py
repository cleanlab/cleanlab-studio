import os.path

from cleanlab_cli.types import ImageFileExtension
from PIL import Image
from pathlib import Path


def image_file_exists(image_filepath: str, dataset_filepath: str) -> bool:
    if os.path.isabs(image_filepath):
        return os.path.exists(image_filepath)
    else:
        dataset_path = Path(dataset_filepath)
        directory_path = dataset_path.parent.absolute()
        abs_image_filepath = os.path.join(directory_path, image_filepath)
        return os.path.exists(abs_image_filepath)


def is_valid_image(filepath: str) -> bool:
    """
    valid == has extension .jpeg or .png and image file can be opened by Pillow
    :param filepath:
    :return:
    """
    try:
        ImageFileExtension(filepath)
        Image.open(filepath)
    except ValueError:
        return False
    except IOError:
        return False
    return True
