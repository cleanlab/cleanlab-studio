import click
import json
import os
from cleanlab_studio.cli.settings import CleanlabSettings
from cleanlab_studio.cli.types import CommandState
from typing import Dict, Any, Optional

FILENAME = "state.json"


class PreviousState:
    def __init__(self) -> None:
        self.state: CommandState = dict(command=None, args=dict())
        self.load_state()

    def load_state(self) -> None:
        fp = self.get_filepath()
        if os.path.exists(fp):
            with open(fp, "r") as f:
                self.state = json.load(f)

    @staticmethod
    def get_filepath() -> str:
        return os.path.join(CleanlabSettings.get_cleanlab_dir(), FILENAME)

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(self.get_filepath()), exist_ok=True)
        with open(self.get_filepath(), "w") as f:
            json.dump(self.state, f)

    def init_state(self, updates: CommandState) -> None:
        self.state = dict(command=None, args=dict())
        self.state["command"] = updates["command"]
        self.state["args"] = updates["args"]
        self.save_state()

    def update_args(self, args_dict: Dict[str, Any]) -> None:
        self.state["args"].update(args_dict)
        self.save_state()

    def get_state(self) -> CommandState:
        return self.state

    def same_command(self, command_name: str, args: Dict[str, Any]) -> bool:
        if self.get("command") == command_name:
            same = all(self.get_arg(k) == args.get(k, None) for k in args)
            return same
        return False

    def get(self, item: str) -> Optional[Any]:
        return self.state.get(item, None)

    def get_arg(self, item: str) -> Optional[Any]:
        return self.state["args"].get(item, None)


previous_state = click.make_pass_decorator(PreviousState, ensure=True)
