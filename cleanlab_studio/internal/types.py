from enum import Enum
from typing import Any, Dict


JSONDict = Dict[str, Any]


class Modality(Enum):
    text = "text"
    tabular = "tabular"
    image = "image"


MODALITIES = [m.value for m in Modality]
