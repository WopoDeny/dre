"""One monotonic deadline shared by every stage and the optional targeted retry."""

from __future__ import annotations

import math
import time
from typing import Callable

from .errors import EngineError, ErrorCode

Clock = Callable[[], int]
# The 45-second conversion budget includes server-side upload, verification,
# workspace cleanup and worker handoff. The additional ten seconds belong to
# transport and response delivery, not to hidden conversion retries.
NORMAL_TARGET_MS = 42_000
INTERNAL_STOP_MS = 45_000
EXTERNAL_SLA_MS = 55_000


def finite_ms(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number of milliseconds.")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite.")
    return float(value)


class DeadlineBudget:
    """Cooperative deadline checks and timeout values, not a thread preemption mechanism."""

    def __init__(self, timeout_ms: float = INTERNAL_STOP_MS, *, clock: Clock = time.monotonic_ns) -> None:
        requested_ms = finite_ms(timeout_ms, "timeout_ms", positive=True)
        self._clock = clock
        self._started_ns = clock()
        self._duration_ns = max(1, int(min(requested_ms, INTERNAL_STOP_MS) * 1_000_000))
        self._deadline_ns = self._started_ns + self._duration_ns

    @property
    def started_ns(self) -> int:
        return self._started_ns

    @property
    def timeout_ms(self) -> float:
        return self._duration_ns / 1_000_000

    @property
    def elapsed_ms(self) -> float:
        return max(0, self._clock() - self._started_ns) / 1_000_000

    @property
    def remaining_ms(self) -> float:
        return max(0, self._deadline_ns - self._clock()) / 1_000_000

    def check(self, *, stage: str | None = None) -> None:
        if self._clock() >= self._deadline_ns:
            raise EngineError(
                ErrorCode.DEADLINE_EXCEEDED,
                "The request deadline has been reached.",
                stage=stage,
                details={"timeout_ms": self.timeout_ms},
            )

    def require_fit(
        self,
        estimated_ms: float,
        *,
        reserve_ms: float = 0,
        stage: str | None = "preflight",
    ) -> None:
        """Reject work before it starts if its remaining estimate and reserve cannot fit."""
        estimate = finite_ms(estimated_ms, "estimated_ms")
        reserve = finite_ms(reserve_ms, "reserve_ms")
        self.check(stage=stage)
        remaining = self.remaining_ms
        if estimate + reserve >= remaining:
            raise EngineError(
                ErrorCode.SLA_REJECTED,
                "The estimated remaining work cannot fit the request deadline.",
                stage=stage,
                details={
                    "estimated_ms": estimate,
                    "reserve_ms": reserve,
                    "remaining_ms": remaining,
                },
            )

    def timeout_seconds(self, *, reserve_ms: float = 0, stage: str | None = None) -> float:
        """Return the current timeout for a bounded operation, preserving a tail reserve."""
        reserve = finite_ms(reserve_ms, "reserve_ms")
        self.check(stage=stage)
        available = self.remaining_ms - reserve
        if available <= 0:
            raise EngineError(
                ErrorCode.SLA_REJECTED,
                "No operation time remains after the required reserve.",
                stage=stage,
                details={"reserve_ms": reserve, "remaining_ms": self.remaining_ms},
            )
        return available / 1_000
