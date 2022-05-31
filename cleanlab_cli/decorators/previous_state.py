import click
from typing import Any, Dict
import json
import os
from cleanlab_cli.settings import CleanlabSettings

FILENAME = "state.json"


class PreviousState:
    def __init__(self):
        self.state = dict(command=None, args=dict())
        self.load_state()

    def load_state(self):
        fp = self.get_filepath()
        if os.path.exists(fp):
            with open(fp, "r") as f:
                self.state = json.load(f)

    @staticmethod
    def get_filepath():
        return os.path.join(CleanlabSettings.get_cleanlab_dir(), FILENAME)

    def save_state(self):
        os.makedirs(os.path.dirname(self.get_filepath()), exist_ok=True)
        with open(self.get_filepath(), "w") as f:
            json.dump(self.state, f)

    def init_state(self, updates: Dict[str, Any]):
        self.state = dict()
        self.state.update(updates)
        self.save_state()

    def update_args(self, args_dict):
        self.state["args"].update(args_dict)
        self.save_state()

    def get_state(self):
        return self.state

    def same_command(self, command_name, args):
        if self.get("command") == command_name:
            same = all(self.get_arg(k) == args.get(k, None) for k in args)
            return same
        return False

    def get(self, item):
        return self.state.get(item, None)

    def get_arg(self, item):
        return self.state["args"].get(item, None)


previous_state = click.make_pass_decorator(PreviousState, ensure=True)
