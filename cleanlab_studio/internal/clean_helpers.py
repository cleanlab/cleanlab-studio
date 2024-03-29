import itertools
import time
import traceback
from typing import List, Literal, Optional

from tqdm import tqdm

from cleanlab_studio.errors import CleansetError, CleansetHandledError, InvalidDatasetError
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import log_internal_error


def poll_cleanset_status(
    api_key: str,
    cleanset_id: str,
    timeout: Optional[float] = None,
    show_cleanset_link: bool = False,
) -> None:
    start_time = time.time()
    res = api.get_cleanset_status(api_key, cleanset_id)
    spinner = itertools.cycle("|/-\\")

    with tqdm(
        total=res["total_steps"],
        desc="Cleanset Progress: \\",
        bar_format="{desc} Step {n_fmt}/{total_fmt}{postfix}",
    ) as pbar:
        while not res["is_ready"] and not res["has_error"]:
            if pbar.total is None and res["total_steps"] is not None:
                pbar.total = res["total_steps"]
                pbar.refresh()

            pbar.set_postfix_str(res["step_description"])
            pbar.update(int(res["step"]) - pbar.n)

            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError("Cleanset not ready before timeout")

            for _ in range(50):
                time.sleep(0.1)
                pbar.set_description_str(f"Cleanset Progress: {next(spinner)}")

            res = api.get_cleanset_status(api_key, cleanset_id)

        if res["is_ready"]:
            pbar.update(pbar.total - pbar.n)
            ready_description = res["step_description"]
            if show_cleanset_link:
                ready_description += (
                    f" View your cleanset at: https://app.cleanlab.ai/cleansets/{cleanset_id}"
                )
            pbar.set_postfix_str(ready_description)
            return

        if res["has_error"]:
            pbar.set_postfix_str(res["step_description"])
            error_message = res["error_message"]
            error_type = res["error_type"]

            if error_type is not None:
                if error_message is None:
                    log_internal_error(
                        "Missing error message for handled error",
                        "\n".join(traceback.format_stack()),
                        api_key,
                    )
                raise CleansetHandledError(error_type=error_type, error_message=error_message)

            raise CleansetError(f"Cleanset {cleanset_id} failed to complete")


def validate_label_column(
    api_key: str,
    dataset_id: str,
    label_column: str,
    modality: Literal["text", "tabular", "image"],
    task_type: Optional[Literal["multi-class", "multi-label", "regression", "unsupervised"]],
    possible_label_columns: List[str],
) -> None:
    if label_column not in possible_label_columns:
        if task_type == "multi-class":
            valid_types = ["string", "integer", "boolean"]
        if task_type == "multi-label":
            valid_types = ["string"]
        if task_type == "regression":
            valid_types = ["float"]

        raise InvalidDatasetError(
            (
                f"Invalid label column: {label_column}. "
                f"{str(task_type).capitalize()} projects require a label column of type {', '.join(valid_types)}. "
                "For details on how to set the schema for your dataset, see the [datasets guide](/guide/concepts/datasets/#schema-updates). "
                "Also ensure that the column has at least 2 unique values."
            )
        )
    if task_type == "multi-class":
        column_diversity = api.check_column_diversity(api_key, dataset_id, label_column)
        if (
            modality != "text"
            and modality != "image"
            and not column_diversity["has_minimal_diversity"]
        ):
            raise InvalidDatasetError(
                "Label column for multi-class projects must have at least 2 unique classes with at least 5 examples each."
            )
    if task_type == "multi-label":
        if not api.is_valid_multilabel_column(api_key, dataset_id, label_column):
            raise InvalidDatasetError(
                'Label column for multi-label projects should be formatted as comma-separated string of labels, i.e. "wearing_hat,has_glasses"'
            )
