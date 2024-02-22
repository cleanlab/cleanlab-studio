from typing import List

# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_DEFAULT_MAX_CONCURRENT_TLM_REQUESTS: int = 16
_MAX_CONCURRENT_TLM_REQUESTS_LIMIT: int = 128
_VALID_TLM_QUALITY_PRESETS: List[str] = ["best", "high", "medium", "low", "base"]
