from cleanlab_studio.errors import EnrichmentProjectError, EnrichmentProjectHandledError
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import log_internal_error


from tqdm import tqdm


import itertools
import time
import traceback


def poll_enrichment_status(
    api_key: str,
    project_id: str,
    job_id: str,
) -> None:
    res = api.get_enrichment_status(api_key, job_id)
    spinner = itertools.cycle("|/-\\")

    with tqdm(
        total=res["total_steps"],
        desc="Enrichment Progress: \\",
        bar_format="{desc} Step {n_fmt}/{total_fmt}{postfix}",
    ) as pbar:
        while not res["is_ready"] and not res["has_error"]:
            if pbar.total is None and res["total_steps"] is not None:
                pbar.total = res["total_steps"]
                pbar.refresh()

            pbar.set_postfix_str(res["step_description"])
            pbar.update(int(res["step"]) - pbar.n)

            for _ in range(50):
                time.sleep(0.1)
                pbar.set_description_str(f"Enrichment Progress: {next(spinner)}")

            res = api.get_enrichment_status(api_key, job_id)

        if res["is_ready"]:
            pbar.update(pbar.total - pbar.n)
            ready_description = res["step_description"]
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
                raise EnrichmentProjectHandledError(
                    error_type=error_type, error_message=error_message
                )

            raise EnrichmentProjectError(f"Project {project_id} failed to complete")


def enrichment_ready(
    api_key: str,
    job_id: str,
) -> bool:
    res = api.get_enrichment_status(api_key, job_id)
    return res["is_ready"] and not res["has_error"]
