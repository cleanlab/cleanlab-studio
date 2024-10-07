from typing import Dict, List, Set, Tuple

# TLM constants
# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_VALID_TLM_QUALITY_PRESETS: List[str] = ["best", "high", "medium", "low", "base"]
_VALID_TLM_MODELS: List[str] = [
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
]
_TLM_DEFAULT_MODEL: str = "gpt-4o-mini"
_TLM_MAX_RETRIES: int = 3  # TODO: finalize this number
_TLM_MAX_TOKEN_RANGE: Dict[str, Tuple[int, int]] = {  # model: (min, max)
    "gpt-3.5-turbo-16k": (64, 4096),
    "gpt-4": (64, 4096),
    "gpt-4o": (64, 4096),
    "gpt-4o-mini": (64, 4096),
    "o1-preview": (64, 4096),
    "claude-3-haiku": (64, 512),
    "claude-3-sonnet": (64, 512),
    "claude-3.5-sonnet": (64, 512),
}
TLM_NUM_CANDIDATE_RESPONSES_RANGE: Tuple[int, int] = (1, 20)  # (min, max)
TLM_NUM_CONSISTENCY_SAMPLES_RANGE: Tuple[int, int] = (0, 20)  # (min, max)
TLM_VALID_LOG_OPTIONS: Set[str] = {"perplexity", "explanation"}
TLM_VALID_GET_TRUSTWORTHINESS_SCORE_KWARGS: Set[str] = {"perplexity"}
