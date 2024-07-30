from typing import List, Tuple, Set

# TLM constants
# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_VALID_TLM_QUALITY_PRESETS: List[str] = ["best", "high", "medium", "low", "base"]
_VALID_TLM_MODELS: List[str] = [
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
]
_TLM_MAX_RETRIES: int = 3  # TODO: finalize this number
TLM_MAX_TOKEN_RANGE: Tuple[int, int] = (64, 512)  # (min, max)
TLM_NUM_CANDIDATE_RESPONSES_RANGE: Tuple[int, int] = (1, 20)  # (min, max)
TLM_NUM_CONSISTENCY_SAMPLES_RANGE: Tuple[int, int] = (0, 20)  # (min, max)
TLM_VALID_LOG_OPTIONS: Set[str] = {"perplexity"}
TLM_VALID_GET_TRUSTWORTHINESS_SCORE_KWARGS: Set[str] = {"perplexity"}
