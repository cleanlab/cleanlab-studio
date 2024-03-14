import asyncio
import contextlib
from types import TracebackType
from typing import Optional, Type

from cleanlab_studio.errors import RateLimitError


class TlmRateHandler(contextlib.AbstractAsyncContextManager):
    """Concurrency handler for TLM queries.

    Implements additive increase / multiplicative decrease congestion control algorithm.
    """

    DEFAULT_CONGESTION_WINDOW: int = 4
    DEFAULT_SLOW_START_THRESHOLD: int = 16

    SLOW_START_INCREASE_FACTOR: int = 2
    ADDITIVE_INCREMENT: int = 1
    MULTIPLICATIVE_DECREASE_FACTOR: int = 2

    def __init__(
        self,
        congestion_window: int = DEFAULT_CONGESTION_WINDOW,
        slow_start_threshold: int = DEFAULT_SLOW_START_THRESHOLD,
    ):
        """Initializes TLM rate handler."""
        self._congestion_window: int = congestion_window
        self._slow_start_threshold = slow_start_threshold

        # create send semaphore and seed w/ initial congestion window
        self._send_semaphore = asyncio.Semaphore(value=self._congestion_window)

    async def __aenter__(self):
        """Acquires send semaphore, blocking until it can be acquired."""
        await self._send_semaphore.acquire()
        return

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback_type: Optional[TracebackType],
    ) -> bool:
        """Handles exiting from rate limit context. Never suppresses exceptions.

        If request succeeded, increase congestion window.
        If request failed due to rate limit error, decrease congestion window.
        Else if request failed for other reason, don't change congestion window, just exit.
        """
        if exc_type is None:
            await self._increase_congestion_window()

        elif exc_type is RateLimitError:
            await self._decrease_congestion_window()

        # release acquired send semaphore from aenter
        self._send_semaphore.release()

        return False

    async def _increase_congestion_window(
        self,
        slow_start_increase_factor: int = SLOW_START_INCREASE_FACTOR,
        additive_increment: int = ADDITIVE_INCREMENT,
    ) -> None:
        """Increases TLM congestion window

        If in slow start, increase is exponential.
        Otherwise, increase is linear.

        After increasing congestion window, notify on send condition with n=increase
        """
        # track previous congestion window size
        prev_congestion_window = self._congestion_window

        # increase congestion window
        if self._congestion_window < self._slow_start_threshold:
            self._congestion_window *= slow_start_increase_factor

        else:
            self._congestion_window += additive_increment

        # release <congestion_window_increase> from send semaphore
        congestion_window_increase = self._congestion_window - prev_congestion_window
        for _ in range(congestion_window_increase):
            self._send_semaphore.release()

    async def _decrease_congestion_window(
        self,
        multiplicative_decrease_factor: int = MULTIPLICATIVE_DECREASE_FACTOR,
    ) -> None:
        """Decreases TLM congestion window, to minimum of 1."""
        if self._congestion_window == 1:
            return

        prev_congestion_window = self._congestion_window
        self._congestion_window //= multiplicative_decrease_factor

        # acquire congestion window decrease from send semaphore
        congestion_window_decrease = self._congestion_window - prev_congestion_window
        for _ in range(congestion_window_decrease):
            async with self._send_semaphore:
                pass
