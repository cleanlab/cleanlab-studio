import click
from typing import Dict
import json
import os
from cleanlab_cli.settings import CleanlabSettings

FILENAME = "state.json"


class PreviousState:
    def __init__(self):
        self.state = dict()

    @staticmethod
    def get_filepath():
        return os.path.join(CleanlabSettings.get_cleanlab_dir(), FILENAME)

    def save_state(self):
        with open(self.get_filepath(), "w") as f:
            json.dump(self.state, f)

    def new_state(self, updates: Dict[str, any]):
        self.state = dict()
        self.state.update(updates)
        self.save_state()

    def update_state(self, updates: Dict[str, any]):
        """Updates and saves state"""
        self.state.update(updates)
        self.save_state()

    def get_state(self):
        if len(self.state) == 0:
            fp = self.get_filepath()
            if os.path.exists(fp):
                with open(fp, "r") as f:
                    self.state = json.load(f)
                    return self.state
            else:
                return {}
        else:
            return self.state

    def same_command(self, command: Dict):
        state = self.get_state()
        if "command" not in state:
            return False
        prev_command = state["command"]
        same = all(prev_command.get(k, None) == command.get(k, None) for k in command)
        return same

    def __getattr__(self, item):
        return self.state.get(item, None)


previous_state = click.make_pass_decorator(PreviousState, ensure=True)
