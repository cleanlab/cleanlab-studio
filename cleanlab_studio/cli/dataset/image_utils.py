import pathlib

from PIL import Image


def get_image_filepath(base_directory: pathlib.Path, image_filepath: pathlib.Path) -> pathlib.Path:
    """Joins base directory with relative or absolute image filepath.

    Note
    ----
    if column_value (image filepath) is an absolute path, base directory will be ignored
    """
    return base_directory.joinpath(image_filepath)


def image_file_exists(image_filepath: pathlib.Path) -> bool:
    return image_filepath.exists()


def image_file_readable(image_filepath: pathlib.Path) -> bool:
    """
    readable == image file can be opened by Pillow
    """
    try:
        Image.open(image_filepath)
        return True

    except IOError:
        return False
