import enum
import itertools
import time
from typing import Optional

from cleanlab_studio.errors import DeploymentError
from cleanlab_studio.internal.api import api
from tqdm import tqdm


class DeploymentStatus(enum.Enum):
    INITIALIZING = "INITIALIZING"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    FAILED = "FAILED"


def poll_deployment_status(api_key: str, model_id: str, timeout: Optional[float] = None) -> None:
    start_time = time.time()
    res = api.get_deployment_status(api_key, model_id)
    spinner = itertools.cycle("|/-\\")

    with tqdm(total=1, desc="Deploying Model: \\", bar_format="{desc}") as pbar:
        while (
            not res == DeploymentStatus.TRAINED.value and not res == DeploymentStatus.FAILED.value
        ):
            pbar.set_description_str(f"Deploying Model: {next(spinner)}")
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError("Model not ready before timeout")

            for _ in range(50):
                time.sleep(0.1)
                pbar.set_description_str(f"Deploying Model: {next(spinner)}")

            res = api.get_deployment_status(api_key, model_id)

            if res == DeploymentStatus.TRAINED.value:
                pbar.set_description_str("Model trained successfully")
                pbar.update(pbar.total - pbar.n)
                return

            if res == DeploymentStatus.FAILED.value:
                pbar.set_description_str("Model training failed")
                raise DeploymentError(f"Model training failed for {model_id}")
