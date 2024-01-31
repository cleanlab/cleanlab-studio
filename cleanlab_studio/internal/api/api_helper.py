import uuid


def check_uuid_well_formed(uuid_string: str, id_name: str) -> None:
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        raise ValueError(
            f"{uuid_string} is not a well-formed {id_name}, please double check and try again."
        )
