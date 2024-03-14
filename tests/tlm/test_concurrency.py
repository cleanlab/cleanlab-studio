import math

import pytest

from cleanlab_studio.errors import RateLimitError
from cleanlab_studio.internal.tlm.concurrency import TlmRateHandler


@pytest.mark.asyncio
async def test_rate_handler_slow_start(tlm_rate_handler: TlmRateHandler) -> None:
    """Tests rate handler increase behavior in slow start.

    Expected behavior:
    - Limiter increases congestion window exponentially up to slow start threshold.
    - Limiter send semaphore value matches congestion window
    """
    # compute number of expected slow start increases
    expected_slow_start_increases = int(
        math.log(
            tlm_rate_handler.DEFAULT_SLOW_START_THRESHOLD,
            tlm_rate_handler.SLOW_START_INCREASE_FACTOR,
        )
        / tlm_rate_handler.DEFAULT_CONGESTION_WINDOW
    )

    # after every rate limiter acquisition, assert:
    # - congestion window *= SLOW_START_INCREASE_FACTOR
    # - congestion window == send_semaphore value
    for i in range(1, expected_slow_start_increases + 1):
        async with tlm_rate_handler:
            pass

        expected_congestion_window = tlm_rate_handler.DEFAULT_CONGESTION_WINDOW * (
            tlm_rate_handler.SLOW_START_INCREASE_FACTOR**i
        )
        assert (
            tlm_rate_handler._congestion_window == expected_congestion_window
        ), "Congestion window is not increased exponentially in slow start"
        assert (
            tlm_rate_handler._send_semaphore._value == tlm_rate_handler._congestion_window
        ), "Send semaphore value does not match congestion window in slow start"


@pytest.mark.asyncio
async def test_rate_handler_additive_increase(
    tlm_rate_handler: TlmRateHandler, num_additive_increases: int = 100
) -> None:
    """Tests rate handler increase behavior in congestion control / additive increase phase.

    Expected behavior:
    - Limiter increases congestion window linearly beyond slow start window
    - Limiter send semaphore value matches congestion window
    """
    # arrange -- skip past slow start phase
    current_limit_value = tlm_rate_handler.DEFAULT_SLOW_START_THRESHOLD
    tlm_rate_handler._congestion_window = current_limit_value
    tlm_rate_handler._send_semaphore._value = current_limit_value

    # after every rate limiter acquisition, assert:
    # - congestion window *= SLOW_START_INCREASE_FACTOR
    # - congestion window == send_semaphore value
    for expected_limit_value in range(
        current_limit_value + 1, num_additive_increases + current_limit_value + 1
    ):
        async with tlm_rate_handler:
            pass

        assert (
            tlm_rate_handler._congestion_window == expected_limit_value
        ), "Congestion window is not increased linearly in congestion control"
        assert (
            tlm_rate_handler._send_semaphore._value == tlm_rate_handler._congestion_window
        ), "Send semaphore value does not match congestion window in congestion control"


@pytest.mark.parametrize("initial_congestion_window", [4, 5, 10, 101])
@pytest.mark.asyncio
async def test_rate_handler_rate_limit_error(
    tlm_rate_handler: TlmRateHandler,
    initial_congestion_window: int,
) -> None:
    """Tests rate handler decrease behavior on a rate limit error.

    Expected behavior:
    - Limiter decreases congestion window multiplicatively
    - Limiter send semaphore value matches congestion window
    - RateLimitError is raised (not suppressed by context manager)
    """
    # arrange -- set current congestion window
    tlm_rate_handler._congestion_window = initial_congestion_window
    tlm_rate_handler._send_semaphore._value = initial_congestion_window

    # acquire rate limit and raise rate limit error, check that:
    # - congestion window is decreased multiplicatively
    # - send semaphore value matches congestion window
    # - rate limit error is raised
    with pytest.raises(RateLimitError):
        async with tlm_rate_handler:
            raise RateLimitError("", 0)

    assert (
        tlm_rate_handler._congestion_window
        == initial_congestion_window // tlm_rate_handler.MULTIPLICATIVE_DECREASE_FACTOR
    ), "Congestion window is not decreased multiplicatively in congestion avoidance"
    assert (
        tlm_rate_handler._send_semaphore._value == tlm_rate_handler._congestion_window
    ), "Send semaphore value does not match congestion window in congestion avoidance"


@pytest.mark.parametrize("initial_congestion_window", [4, 5, 10, 101])
@pytest.mark.asyncio
async def test_rate_handler_non_rate_limit_error(
    tlm_rate_handler: TlmRateHandler,
    initial_congestion_window: int,
) -> None:
    """Tests rate handler decrease behavior on a NON rate limit error.

    Expected behavior:
    - Limiter congestion window stays the same
    - Limiter send semaphore value matches congestion window
    - error is raised (not suppressed by context manager)
    """
    # arrange -- set current congestion window
    tlm_rate_handler._congestion_window = initial_congestion_window
    tlm_rate_handler._send_semaphore._value = initial_congestion_window

    # acquire rate limit and raise rate limit error, check that:
    # - congestion window is decreased multiplicatively
    # - send semaphore value matches congestion window
    # - rate limit error is raised
    with pytest.raises(ValueError):
        async with tlm_rate_handler:
            raise ValueError

    assert (
        tlm_rate_handler._congestion_window == initial_congestion_window
    ), "Congestion window is kept same for non rate limit error"
    assert (
        tlm_rate_handler._send_semaphore._value == tlm_rate_handler._congestion_window
    ), "Send semaphore value does not match congestion window after non rate limit error"
