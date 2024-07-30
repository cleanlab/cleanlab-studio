"""
Cleanlab's Trustworthy Language Model (TLM) is a large language model that gives more reliable answers and quantifies its uncertainty in these answers.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.TLM()`](/reference/python/studio/#method-tlm) to instantiate a [TLM](#class-tlm) object, and then you can use the methods like [`prompt()`](#method-prompt) and [`get_trustworthiness_score()`](#method-get_trustworthiness_score) documented on this page.

The [Trustworthy Language Model tutorial](/tutorials/tlm/) further explains TLM and its use cases.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Coroutine, List, Optional, Union, cast, Sequence, Any, Dict
from tqdm.asyncio import tqdm_asyncio
import numpy as np

import aiohttp
from typing_extensions import NotRequired, TypedDict  # for Python <3.11 with (Not)Required

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.tlm.concurrency import TlmRateHandler
from cleanlab_studio.internal.tlm.validation import (
    validate_tlm_prompt,
    validate_tlm_try_prompt,
    validate_tlm_prompt_response,
    validate_try_tlm_prompt_response,
    validate_tlm_options,
    process_response_and_kwargs,
)
from cleanlab_studio.internal.types import TLMQualityPreset
from cleanlab_studio.errors import ValidationError
from cleanlab_studio.internal.constants import (
    _VALID_TLM_QUALITY_PRESETS,
    _TLM_MAX_RETRIES,
)


class TLM:
    """
    Represents a Trustworthy Language Model (TLM) instance, which is bound to a Cleanlab Studio account.

    The TLM object can be used as a drop-in replacement for an LLM, or, for estimating trustworthiness scores for arbitrary text prompt/response pairs.

    For advanced use, TLM offers configuration options. The documentation below summarizes these options, and more details are explained in the [TLM tutorial](/tutorials/tlm).

    ** The TLM object is not meant to be constructed directly.** Instead, use the [`Studio.TLM()`](../studio/#method-tlm)
    method to configure and instantiate a TLM object.
    After you've instantiated the TLM object using [`Studio.TLM()`](../studio/#method-tlm), you can use the instance methods documented on this page. Possible arguments for `Studio.TLM()` are documented below.

    Args:
        quality_preset (TLMQualityPreset, default = "medium"): An optional preset configuration to control the quality of TLM responses and trustworthiness scores vs. runtimes/costs. TLMQualityPreset is a string specifying one of the supported presets: "best", "high", "medium", "low", "base".

            The "best" and "high" presets return improved LLM responses,
            with "best" also returning more reliable trustworthiness scores than "high".
            The "medium" and "low" presets return standard LLM responses along with associated trustworthiness scores,
            with "medium" producing more reliable trustworthiness scores than low.
            The "base" preset will not return any trustworthiness score, just a standard LLM response, and is similar to directly using your favorite LLM API.

            Higher presets have increased runtime and cost (and may internally consume more tokens).
            Reduce your preset if you see token-limit errors.
            Details about each present are in the documentation for [TLMOptions](#class-tlmoptions).
            Avoid using "best" or "high" presets if you primarily want to get trustworthiness scores, and are less concerned with improving LLM responses.
            These presets have higher runtime/cost and are optimized to return more accurate LLM outputs, but not necessarily more reliable trustworthiness scores.

        options (TLMOptions, optional): a typed dict of advanced configuration options.
        Available options (keys in this dict) include "model", "max_tokens", "num_candidate_responses", "num_consistency_samples", "use_self_reflection".
        For more details about the options, see the documentation for [TLMOptions](#class-tlmoptions).
        If specified, these override any settings from the choice of `quality_preset`.

        timeout (float, optional): timeout (in seconds) to apply to each TLM prompt.
        If a batch of data is passed in, the timeout will be applied to each individual item in the batch.
        If a result is not produced within the timeout, a TimeoutError will be raised. Defaults to None, which does not apply a timeout.

        verbose (bool, optional): whether to print outputs during execution, i.e., whether to show a progress bar when TLM is prompted with batches of data.
        If None, this will be determined automatically based on whether the code is running in an interactive environment such as a Jupyter notebook.
    """

    def __init__(
        self,
        api_key: str,
        quality_preset: TLMQualityPreset,
        *,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """Use `Studio.TLM()` instead of this method to initialize a TLM.
        lazydocs: ignore
        """
        self._api_key = api_key

        if quality_preset not in _VALID_TLM_QUALITY_PRESETS:
            raise ValidationError(
                f"Invalid quality preset {quality_preset} -- must be one of {_VALID_TLM_QUALITY_PRESETS}"
            )

        self._return_log = False
        if options is not None:
            validate_tlm_options(options)
            if "log" in options.keys():
                self._return_log = True

        if timeout is not None and not (isinstance(timeout, int) or isinstance(timeout, float)):
            raise ValidationError("timeout must be a integer or float value")

        if verbose is not None and not isinstance(verbose, bool):
            raise ValidationError("verbose must be a boolean value")

        is_notebook_flag = is_notebook()

        self._quality_preset = quality_preset
        self._options = options
        self._timeout = timeout if timeout is not None and timeout > 0 else None
        self._verbose = verbose if verbose is not None else is_notebook_flag

        if is_notebook_flag:
            import nest_asyncio

            nest_asyncio.apply()

        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
        self._rate_handler = TlmRateHandler()

    async def _batch_prompt(
        self,
        prompts: Sequence[str],
        capture_exceptions: bool = False,
    ) -> Union[List[TLMResponse], List[Optional[TLMResponse]]]:
        """Run a batch of prompts through TLM and get responses/scores for each prompt in the batch. The list returned will have the same length as the input list.

        Args:
            prompts (List[str]): list of prompts to run
            capture_exceptions (bool): if ``True``, the returned list will contain ``None`` in place of the response for any errors or timeout when processing a particular prompt from the batch.
                If ``False``, this entire method will raise an exception if TLM fails to produce a result for any prompt in the batch.

        Returns:
            Union[List[TLMResponse], List[Optional[TLMResponse]]]: TLM responses/scores for each prompt (in supplied order)
        """
        if capture_exceptions:
            per_query_timeout, per_batch_timeout = self._timeout, None
        else:
            per_query_timeout, per_batch_timeout = None, self._timeout

        # run batch of TLM
        tlm_responses = await self._batch_async(
            [
                self._prompt_async(
                    prompt,
                    timeout=per_query_timeout,
                    capture_exceptions=capture_exceptions,
                    batch_index=batch_index,
                )
                for batch_index, prompt in enumerate(prompts)
            ],
            per_batch_timeout,
        )

        if capture_exceptions:
            return cast(List[Optional[TLMResponse]], tlm_responses)

        return cast(List[TLMResponse], tlm_responses)

    async def _batch_get_trustworthiness_score(
        self,
        prompts: Sequence[str],
        responses: Sequence[Dict[str, Any]],
        capture_exceptions: bool = False,
    ) -> Union[TLMBatchScoreResponse, TLMOptionalBatchScoreResponse]:
        """Run batch of TLM get trustworthiness score.

        capture_exceptions behavior:
        - If true, the list will contain None in place of the response for any errors or timeout processing some inputs.
        - Otherwise, the method will raise an exception for any errors or timeout processing some inputs.

        capture_exceptions interaction with timeout:
        - If true, timeouts are applied on a per-query basis (i.e. some queries may succeed while others fail)
        - If false, a single timeout is applied to the entire batch (i.e. all queries will fail if the timeout is reached)

        Args:
            prompts (Sequence[str]): list of prompts to run get trustworthiness score for
            responses (Sequence[str]): list of responses to run get trustworthiness score for
            capture_exceptions (bool): if should return None in place of the response for any errors or timeout processing some inputs

        Returns:
            Union[TLMBatchScoreResponse, TLMOptionalBatchScoreResponse]: TLM trustworthiness score for each prompt (in supplied order)
        """
        if capture_exceptions:
            per_query_timeout, per_batch_timeout = self._timeout, None
        else:
            per_query_timeout, per_batch_timeout = None, self._timeout

        # run batch of TLM get trustworthiness score
        tlm_responses = await self._batch_async(
            [
                self._get_trustworthiness_score_async(
                    prompt,
                    response,
                    timeout=per_query_timeout,
                    capture_exceptions=capture_exceptions,
                    batch_index=batch_index,
                )
                for batch_index, (prompt, response) in enumerate(zip(prompts, responses))
            ],
            per_batch_timeout,
        )

        if capture_exceptions:
            return cast(TLMOptionalBatchScoreResponse, tlm_responses)

        return cast(TLMBatchScoreResponse, tlm_responses)

    async def _batch_async(
        self,
        tlm_coroutines: Sequence[Coroutine[None, None, Union[TLMResponse, TLMScoreResponse, None]]],
        batch_timeout: Optional[float] = None,
    ) -> Sequence[Union[TLMResponse, float, None]]:
        """Runs batch of TLM queries.

        Args:
            tlm_coroutines (List[Coroutine[None, None, Union[TLMResponse, float, None]]]): list of query coroutines to run, returning TLM responses or trustworthiness scores (or None if capture_exceptions is True)
            batch_timeout (Optional[float], optional): timeout (in seconds) to run all queries, defaults to None (no timeout)

        Returns:
            Sequence[Union[TLMResponse, float, None]]: list of coroutine results, with preserved order
        """
        tlm_query_tasks = [asyncio.create_task(tlm_coro) for tlm_coro in tlm_coroutines]

        if self._verbose:
            gather_task = tqdm_asyncio.gather(
                *tlm_query_tasks,
                total=len(tlm_query_tasks),
                desc="Querying TLM...",
                bar_format="{desc} {percentage:3.0f}%|{bar}|",
            )
        else:
            gather_task = asyncio.gather(*tlm_query_tasks)

        wait_task = asyncio.wait_for(gather_task, timeout=batch_timeout)
        try:
            return cast(
                Sequence[Union[TLMResponse, float, None]],
                await wait_task,
            )
        except Exception:
            # if exception occurs while awaiting batch results, cancel remaining tasks
            for query_task in tlm_query_tasks:
                query_task.cancel()

            # await remaining tasks to ensure they are cancelled
            await asyncio.gather(*tlm_query_tasks, return_exceptions=True)

            raise

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
        /,
    ) -> Union[TLMResponse, List[TLMResponse]]:
        """
        Gets response and trustworthiness score for any text input.

        This method prompts the TLM with the given prompt(s), producing completions (like a standard LLM)
        but also provides trustworthiness scores quantifying the quality of the output.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the language model.
                Providing a batch of many prompts here will be faster than calling this method on each prompt separately.
        Returns:
            TLMResponse | List[TLMResponse]: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified).
                Use it if you want strict error handling and immediate notification of any exceptions/timeouts.

                If running this method on a big batch of prompts: you might lose partially completed results if TLM fails on any one of them.
                To avoid losing partial results for the prompts that TLM did not fail on,
                you can either call this method on smaller batches of prompts at a time
                (and save intermediate results between batches), or use the [`try_prompt()`](#method-try_prompt) method instead.
        """
        validate_tlm_prompt(prompt)

        if isinstance(prompt, str):
            return cast(
                TLMResponse,
                self._event_loop.run_until_complete(
                    self._prompt_async(prompt, timeout=self._timeout, capture_exceptions=False),
                ),
            )

        return cast(
            List[TLMResponse],
            self._event_loop.run_until_complete(
                self._batch_prompt(prompt, capture_exceptions=False),
            ),
        )

    def try_prompt(
        self,
        prompt: Sequence[str],
        /,
    ) -> List[Optional[TLMResponse]]:
        """
        Gets response and trustworthiness score for any batch of prompts,
        handling any failures (errors of timeouts) by returning None in place of the failures.

        The list returned will have the same length as the input list, if there are any
        failures (errors or timeout) processing some inputs, the list will contain None in place of the response.

        This is the recommended way to get TLM responses and trustworthiness scores for big datasets of many prompts,
        where some individual TLM responses within the dataset may fail. It ensures partial results are not lost.

        Args:
            prompt (Sequence[str]): list of multiple prompts for the TLM
        Returns:
            List[Optional[TLMResponse]]: list of [TLMResponse](#class-tlmresponse) objects containing the response and trustworthiness score.
                The returned list will always have the same length as the input list.
                In case of TLM failure on any prompt (due to timeouts or other errors),
                the return list will contain None in place of the TLM response for that failed prompt.
                Use this to obtain TLM results for as many prompts as possible,
                but you might miss out on certain error messages.
                If you prefer to be notified immediately about any errors or timeouts when running many prompts,
                use the [`prompt()`](#method-prompt) method instead.
        """
        validate_tlm_try_prompt(prompt)

        return cast(
            List[Optional[TLMResponse]],
            self._event_loop.run_until_complete(
                self._batch_prompt(prompt, capture_exceptions=True),
            ),
        )

    async def prompt_async(
        self,
        prompt: Union[str, Sequence[str]],
        /,
    ) -> Union[TLMResponse, List[TLMResponse]]:
        """
        Asynchronously get response and trustworthiness score for any text input from TLM.
        This method is similar to the [`prompt()`](#method-prompt) method but operates asynchronously,
        allowing for non-blocking concurrent operations.

        Use this method if prompts are streaming in one at a time, and you want to return results
        for each one as quickly as possible, without the TLM execution of any one prompt blocking the execution of the others.
        Asynchronous methods do not block until completion, so you will need to fetch the results yourself.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
        Returns:
            TLMResponse | List[TLMResponse]: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified).
        """
        validate_tlm_prompt(prompt)

        async with aiohttp.ClientSession() as session:
            if isinstance(prompt, str):
                tlm_response = await self._prompt_async(
                    prompt, session, timeout=self._timeout, capture_exceptions=False
                )
                return cast(TLMResponse, tlm_response)

            return cast(
                List[TLMResponse],
                await self._batch_prompt(prompt, capture_exceptions=False),
            )

    async def _prompt_async(
        self,
        prompt: str,
        client_session: Optional[aiohttp.ClientSession] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,
        batch_index: Optional[int] = None,
    ) -> Optional[TLMResponse]:
        """
        Private asynchronous method to get response and trustworthiness score from TLM.

        Args:
            prompt (str): prompt for the TLM
            client_session (aiohttp.ClientSession, optional): async HTTP session to use for TLM query. Defaults to None (creates a new session).
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions: if should return None in place of the response for any errors
            batch_index: index of the prompt in the batch, used for error messages
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
        """

        try:
            response_json = await asyncio.wait_for(
                api.tlm_prompt(
                    self._api_key,
                    prompt,
                    self._quality_preset,
                    self._options,
                    self._rate_handler,
                    client_session,
                    batch_index=batch_index,
                    retries=_TLM_MAX_RETRIES,
                ),
                timeout=timeout,
            )
        except Exception as e:
            if capture_exceptions:
                return None
            raise e

        tlm_response = {
            "response": response_json["response"],
            "trustworthiness_score": response_json["confidence_score"],
        }

        if self._return_log:
            tlm_response["log"] = response_json["log"]

        return cast(TLMResponse, tlm_response)

    def get_trustworthiness_score(
        self,
        prompt: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> Union[TLMScoreResponse, TLMBatchScoreResponse]:
        """Computes trustworthiness score for arbitrary given prompt-response pairs.

        Args:
            prompt (str | Sequence[str]): prompt (or list of prompts) for the TLM to evaluate
            response (str | Sequence[str]): existing response (or list of responses) associated with the input prompts.
                These can be from any LLM or human-written responses.
        Returns:
            float | List[float]: float or list of floats (if multiple prompt-responses were provided) corresponding
                to the TLM's trustworthiness score.
                The score quantifies how confident TLM is that the given response is good for the given prompt.
                If running on many prompt-response pairs simultaneously:
                this method will raise an exception if any TLM errors or timeouts occur.
                Use it if strict error handling and immediate notification of any exceptions/timeouts is preferred.
                You will lose any partial results if an exception is raised.
                If saving partial results is important, you can call this method on smaller batches of prompt-response pairs at a time
                (and save intermediate results) or use the [`try_get_trustworthiness_score()`](#method-try_get_trustworthiness_score) method instead.
        """
        validate_tlm_prompt_response(prompt, response)
        processed_response = process_response_and_kwargs(response, kwargs)

        if isinstance(prompt, str) and isinstance(processed_response, dict):
            return cast(
                TLMScoreResponse,
                self._event_loop.run_until_complete(
                    self._get_trustworthiness_score_async(
                        prompt,
                        processed_response,
                        timeout=self._timeout,
                        capture_exceptions=False,
                    )
                ),
            )

        assert isinstance(prompt, Sequence) and isinstance(processed_response, Sequence)

        return cast(
            TLMBatchScoreResponse,
            self._event_loop.run_until_complete(
                self._batch_get_trustworthiness_score(
                    prompt, processed_response, capture_exceptions=False
                )
            ),
        )

    def try_get_trustworthiness_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
        **kwargs: Any,
    ) -> TLMOptionalBatchScoreResponse:
        """Gets trustworthiness score for batches of many prompt-response pairs.

        The list returned will have the same length as the input list, if TLM hits any
        errors or timeout processing certain inputs, the list will contain None
        in place of the TLM score for this failed input.

        This is the recommended way to get TLM trustworthiness scores for big datasets,
        where some individual TLM calls within the dataset may fail. It will ensure partial results are not lost.

        Args:
            prompt (Sequence[str]): list of prompts for the TLM to evaluate
            response (Sequence[str]): list of existing responses corresponding to the input prompts (from any LLM or human-written)
        Returns:
            List[float]: list of floats corresponding to the TLM's trustworthiness score.
                The score quantifies how confident TLM is that the given response is good for the given prompt.
                The returned list will always have the same length as the input list.
                In case of TLM error or timeout on any prompt-response pair,
                the returned list will contain None in place of the trustworthiness score.
                Use this method if you prioritize obtaining results for as many inputs as possible,
                however you might miss out on certain error messages.
                If you prefer to be notified immediately about any errors or timeouts,
                use the [`get_trustworthiness_score()`](#method-get_trustworthiness_score) method instead.
        """
        validate_try_tlm_prompt_response(prompt, response)
        processed_response = process_response_and_kwargs(response, kwargs)

        assert isinstance(processed_response, list)

        return cast(
            TLMOptionalBatchScoreResponse,
            self._event_loop.run_until_complete(
                self._batch_get_trustworthiness_score(
                    prompt,
                    processed_response,
                    capture_exceptions=True,
                )
            ),
        )

    async def get_trustworthiness_score_async(
        self,
        prompt: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> Union[TLMScoreResponse, List[float], List[TLMScore]]:
        """Asynchronously gets trustworthiness score for prompt-response pairs.
        This method is similar to the [`get_trustworthiness_score()`](#method-get_trustworthiness_score) method but operates asynchronously,
        allowing for non-blocking concurrent operations.

        Use this method if prompt-response pairs are streaming in, and you want to return TLM scores
        for each pair as quickly as possible, without the TLM scoring of any one pair blocking the scoring of the others.
        Asynchronous methods do not block until completion, so you will need to fetch the results yourself.

        Args:
            prompt (str | Sequence[str]): prompt (or list of prompts) for the TLM to evaluate
            response (str | Sequence[str]): response (or list of responses) corresponding to the input prompts
        Returns:
            float | List[float]: float or list of floats (if multiple prompt-responses were provided) corresponding
                to the TLM's trustworthiness score.
                The score quantifies how confident TLM is that the given response is good for the given prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified).
        """
        validate_tlm_prompt_response(prompt, response)
        processed_response = process_response_and_kwargs(response, kwargs)

        async with aiohttp.ClientSession() as session:
            if isinstance(prompt, str) and isinstance(processed_response, str):
                trustworthiness_score = await self._get_trustworthiness_score_async(
                    prompt,
                    processed_response,
                    session,
                    timeout=self._timeout,
                    capture_exceptions=False,
                )
                return cast(TLMScoreResponse, trustworthiness_score)

            assert isinstance(prompt, Sequence) and isinstance(processed_response, Sequence)

            return cast(
                TLMBatchScoreResponse,
                await self._batch_get_trustworthiness_score(
                    prompt, processed_response, capture_exceptions=False
                ),
            )

    async def _get_trustworthiness_score_async(
        self,
        prompt: str,
        response: Dict[str, Any],
        client_session: Optional[aiohttp.ClientSession] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,
        batch_index: Optional[int] = None,
    ) -> Optional[TLMScoreResponse]:
        """Private asynchronous method to get trustworthiness score for prompt-response pairs.

        Args:
            prompt: prompt for the TLM to evaluate
            response: response corresponding to the input prompt
            client_session: async HTTP session to use for TLM query. Defaults to None.
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions: if should return None in place of the response for any errors
            batch_index: index of the prompt in the batch, used for error messages
        Returns:
            float corresponding to the TLM's trustworthiness score

        """
        if self._quality_preset == "base":
            raise ValidationError(
                "Cannot get trustworthiness score with `base` quality_preset -- choose a higher preset."
            )

        try:
            response_json = await asyncio.wait_for(
                api.tlm_get_confidence_score(
                    self._api_key,
                    prompt,
                    response,
                    self._quality_preset,
                    self._options,
                    self._rate_handler,
                    client_session,
                    batch_index=batch_index,
                    retries=_TLM_MAX_RETRIES,
                ),
                timeout=timeout,
            )

            if self._return_log:
                return {
                    "trustworthiness_score": response_json["confidence_score"],
                    "log": response_json["log"],
                }

            return cast(float, response_json["confidence_score"])

        except Exception as e:
            if capture_exceptions:
                return None
            raise e


class TLMResponse(TypedDict):
    """A typed dict containing the response, trustworthiness score and additional logs from the Trustworthy Language Model.

    Attributes:
        response (str): text response from the Trustworthy Language Model.

        trustworthiness_score (float, optional): score between 0-1 corresponding to the trustworthiness of the response.
        A higher score indicates a higher confidence that the response is correct/trustworthy. The trustworthiness score
        is omitted if TLM is run with quality preset "base".

        log (dict, optional): additional logs and metadata returned from the LLM call only if the `log` key was specified in TLMOptions.
    """

    response: str
    trustworthiness_score: Optional[float]
    log: NotRequired[Dict[str, Any]]


class TLMScore(TypedDict):
    """A typed dict containing the trustworthiness score and additional logs from the Trustworthy Language Model.

    This dictionary is similar to TLMResponse, except it does not contain the response key.
    """

    trustworthiness_score: Optional[float]
    log: NotRequired[Dict[str, Any]]


TLMScoreResponse = Union[float, TLMScore]
TLMBatchScoreResponse = Union[List[float], List[TLMScore]]
TLMOptionalBatchScoreResponse = Union[List[Optional[float]], List[Optional[TLMScore]]]


class TLMOptions(TypedDict):
    """Typed dict containing advanced configuration options for the Trustworthy Language Model.
    Many of these configurations are automatically determined by the quality preset selected
    (see the arguments in the TLM [initialization method](../studio#method-tlm) to learn more about quality presets).
    Specifying custom values here will override any default values from the quality preset.

    For all options described below, higher/more expensive settings will lead to longer runtimes and may consume more tokens internally.
    The high token cost might make it such that you are not able to run long prompts (or prompts with long responses) in your account,
    unless your token limits are increased. If you are hit token limit issues, try using lower/less expensive settings
    to be able to run longer prompts/responses.

    The default values corresponding to each quality preset (specified when instantiating [`Studio.TLM()`](../studio/#method-tlm)) are:
    - **best:** `num_candidate_responses` = 6, `num_consistency_samples` = 8, `use_self_reflection` = True. This preset will improve LLM responses.
    - **high:** `num_candidate_responses` = 4, `num_consistency_samples` = 8, `use_self_reflection` = True. This preset will improve LLM responses.
    - **medium:** `num_candidate_responses` = 1, `num_consistency_samples` = 8, `use_self_reflection` = True.
    - **low:** `num_candidate_responses` = 1, `num_consistency_samples` = 4, `use_self_reflection` = True.
    - **base:** `num_candidate_responses` = 1, `num_consistency_samples` = 0, `use_self_reflection` = False. This preset is equivalent to a regular LLM call.

    By default, the TLM is set to the "medium" quality preset. The default `model` used is "gpt-4o-mini", and `max_tokens` is 512 for all quality presets.
    You can set custom values for these arguments regardless of the quality preset specified.

    Args:
        model (str, default = "gpt-4o-mini"): underlying LLM to use (better models will yield better results).
        Models currently supported include "gpt-4o-mini", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4o", "claude-3-haiku".

        max_tokens (int, default = 512): the maximum number of tokens to generate in the TLM response.
        This number will impact the maximum number of tokens you will see in the output response, and also the number of tokens
        that can be generated internally within the TLM (to estimate the trustworthiness score).
        Higher values here can produce better (more reliable) TLM responses and trustworthiness scores, but at higher costs/runtimes.
        If you are experiencing token limit errors while using the TLM (especially on higher quality presets), consider lowering this number.
        This parameter must be between 64 and 512.

        num_candidate_responses (int, default = 1): how many alternative candidate responses are internally generated by TLM.
        TLM scores the trustworthiness of each candidate response, and then returns the most trustworthy one.
        Higher values here can produce better (more accurate) responses from the TLM, but at higher costs/runtimes (and internally consumes more tokens).
        This parameter must be between 1 and 20.
        When it is 1, TLM simply returns a standard LLM response and does not attempt to improve it.

        num_consistency_samples (int, default = 8): the amount of internal sampling to evaluate LLM-response-consistency.
        This consistency forms a big part of the returned trustworthiness score, helping quantify the epistemic uncertainty associated with
        strange prompts or prompts that are too vague/open-ended to receive a clearly defined 'good' response.
        Higher values here produce better (more reliable) TLM trustworthiness scores, but at higher costs/runtimes.
        This parameter must be between 0 and 20.

        use_self_reflection (bool, default = `True`): whether the LLM is asked to self-reflect upon the response it
        generated and self-evaluate this response.
        This self-reflection forms a big part of the trustworthiness score, helping quantify aleatoric uncertainty associated with challenging prompts
        and helping catch answers that are obviously incorrect/bad for a prompt asking for a well-defined answer that LLMs should be able to handle.
        Setting this to False disables the use of self-reflection and may produce worse TLM trustworthiness scores, but will reduce costs/runtimes.

        log (List[str], default = None): optionally specify additional logs or metadata to return.
    """

    model: NotRequired[str]
    max_tokens: NotRequired[int]
    num_candidate_responses: NotRequired[int]
    num_consistency_samples: NotRequired[int]
    use_self_reflection: NotRequired[bool]
    log: NotRequired[List[str]]


def is_notebook() -> bool:
    """Returns True if running in a notebook, False otherwise.

    lazydocs: ignore
    """
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True

        return False
    except:
        return False
