from typing import List

# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_DEFAULT_MAX_CONCURRENT_TLM_REQUESTS: int = 16
_MAX_CONCURRENT_TLM_REQUESTS_LIMIT: int = 128
_VALID_TLM_QUALITY_PRESETS: List[str] = ["best", "high", "medium", "low", "base"]
_VALID_TLM_MODELS: List[str] = ["gpt-3.5-turbo-16k", "gpt-4"]
_MUTABLE_PARAMS_SET_AT_INIT: List[str] = [
    "model"
]  # params that can be set at init of TLM object but also specified in the options dictionary during prompting
