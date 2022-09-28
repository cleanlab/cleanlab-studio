from cleanlab_cli.types import ImageFileExtension
from PIL import Image


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
