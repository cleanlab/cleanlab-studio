import time
from typing import Optional
from tqdm import tqdm

from cleanlab_studio.internal.api import api

READY_STATUS = "6"
ERROR_STATUS = "-1"
FOLD_STATUS = "2"


def get_latest_cleanset_id_when_ready(api_key: str, project_id: str) -> Optional[str]:
    status = api.get_project_status(api_key, project_id)
    if status != READY_STATUS:
        with tqdm(
            total=int(READY_STATUS) - 1,
            desc="Cleanset Progress: \\",
            bar_format="{desc} Step {n_fmt}/{total_fmt}",
        ) as pbar:
            while status != READY_STATUS:
                if status == ERROR_STATUS:
                    print("Cleanset Error")
                    return None
                if status.startswith("FOLD"):
                    pbar.update(int(FOLD_STATUS) - pbar.n)
                else:
                    pbar.update(int(status) - pbar.n)
                for c in "|/-\\":
                    time.sleep(0.1)
                    pbar.set_description_str(f"Cleanset Progress: {c}")
                status = api.get_project_status(api_key, project_id)
            pbar.update(pbar.total - pbar.n)
    return api.get_latest_cleanset_id(api_key, project_id)
