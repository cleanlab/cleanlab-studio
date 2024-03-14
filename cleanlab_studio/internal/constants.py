from typing import List, Tuple

# TLM constants
# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_VALID_TLM_QUALITY_PRESETS: List[str] = ["best", "high", "medium", "low", "base"]
_VALID_TLM_MODELS: List[str] = ["gpt-3.5-turbo-16k", "gpt-4"]
_TLM_MAX_RETRIES: int = 10  # TODO: finalize this number
TLM_MAX_TOKEN_RANGE: Tuple[int, int] = (64, 512)  # (min, max)
TLM_NUM_CANDIDATE_RESPONSES_RANGE: Tuple[int, int] = (1, 100)  # (min, max)
TLM_NUM_CONSISTENCY_SAMPLES_RANGE: Tuple[int, int] = (0, 50)  # (min, max)
