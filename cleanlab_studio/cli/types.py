from typing import Dict, Optional, TypedDict


class CommandState(TypedDict):
    command: Optional[str]
    args: Dict[str, Optional[str]]


