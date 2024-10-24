from __future__ import annotations

import asyncio
from typing import Any, Optional, cast

import json
import aiohttp
from cleanlab_studio.studio.trustworthy_language_model import (
    TLM,
    TLMOptions,
    TLMResponse,
    handle_tlm_exceptions,
)
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.constants import (
    _TLM_MAX_RETRIES,
)
from cleanlab_studio.internal.types import TLMQualityPreset


class TLMOpenAI(TLM):
    def __init__(
        self,
        api_key: str,
        quality_preset: TLMQualityPreset,
        *,
        options: Optional[TLMOptions] = None,
        response_format: Optional[Any] = None,  # TODO: change type
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        # TODO: figure out which validations diverge
        super().__init__(api_key, quality_preset, options=options, timeout=timeout, verbose=verbose)

        self._response_format = response_format
        self._response_format_is_pydantic_object = None
        self._response_format_json = None

        self._openai_args = {}

        if self._response_format is not None:
            try:
                from openai.lib._parsing import type_to_response_format_param
            except ImportError:
                raise ImportError(
                    "Cannot import openai which is required to use TLMOpenAI. "
                    "Please install it using `pip install openai` and try again."
                )

            self._response_format_is_pydantic_object = not isinstance(self._response_format, dict)

            # TODO: could implement ourselves
            self._response_format_json = type_to_response_format_param(self._response_format)

            self._openai_args["response_format"] = self._response_format_json

    @handle_tlm_exceptions("TLMResponse")
    async def _prompt_async(
        self,
        prompt: str,
        client_session: Optional[aiohttp.ClientSession] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,
        batch_index: Optional[int] = None,
    ) -> TLMResponse:
        """
        TODO
        """
        response_json = await asyncio.wait_for(
            api.tlm_openai_prompt(
                self._api_key,
                prompt,
                self._quality_preset,
                self._options,
                self._openai_args,
                self._rate_handler,
                client_session,
                batch_index=batch_index,
                retries=_TLM_MAX_RETRIES,
            ),
            timeout=timeout,
        )

        # TODO: error handling
        response = response_json["response"]
        if self._response_format is not None:
            if self._response_format_is_pydantic_object:
                response = self._response_format(**json.loads(response))
            else:
                response = json.loads(response)

        tlm_response = {
            "response": response,
            "trustworthiness_score": response_json["confidence_score"],
        }

        if self._return_log:
            tlm_response["log"] = response_json["log"]

        # TODO: wrong typing here (need to update TLMResponse)
        return cast(TLMResponse, tlm_response)
