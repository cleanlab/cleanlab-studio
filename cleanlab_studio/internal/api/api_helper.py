import uuid

from cleanlab_studio.errors import InvalidCsvFilename, InvalidUUIDError


def check_uuid_well_formed(uuid_string: str, id_name: str) -> None:
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        raise InvalidUUIDError(
            f"{uuid_string} is not a well-formed {id_name}, please double check and try again."
        )


def check_valid_csv_filename(filename: str) -> None:
    if not filename.lower().endswith(".csv"):
        raise InvalidCsvFilename(f"{filename} is not a valid csv filename.")
