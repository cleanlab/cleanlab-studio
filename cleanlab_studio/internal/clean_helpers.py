import itertools
import time
from typing import Optional

from tqdm import tqdm

from cleanlab_studio.errors import CleansetRunError
from cleanlab_studio.internal.api import api


def poll_cleanset_status(api_key: str, cleanset_id: str, timeout: Optional[float] = None) -> None:
    start_time = time.time()
    res = api.get_cleanset_status(api_key, cleanset_id)
    spinner = itertools.chain("|/-\\")

    with tqdm(
        total=res["total_steps"],
        desc="Cleanset Progress: \\",
        bar_format="{desc} Step {n_fmt}/{total_fmt}{postfix}",
    ) as pbar:
        while not res["is_ready"] and not res["has_error"]:
            for _ in range(50):
                time.sleep(0.1)
                pbar.set_description_str(f"Cleanset Progress: {next(spinner)}")

            pbar.set_postfix_str(res["step_description"])
            pbar.update(int(res["step"]) - pbar.n)
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError("Cleanset not ready before timeout")

            res = api.get_cleanset_status(api_key, cleanset_id)

        if res["is_ready"]:
            pbar.update(pbar.total - pbar.n)
            pbar.set_postfix_str(res["step_description"])
            return

        if res["has_error"]:
            pbar.set_postfix_str(res["step_description"])
            raise CleansetRunError(f"Cleanset {cleanset_id} failed to complete")

    return
