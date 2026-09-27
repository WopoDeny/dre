"""Stable, JSON-safe errors without document content or exception tracebacks."""

from __future__ import annotations

import math
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

JSONScalar = str | int | float | bool | None


class ErrorCode(StrEnum):
    INVALID_PDF = "INVALID_PDF"
    DOCUMENT_TOO_EXPENSIVE = "DOCUMENT_TOO_EXPENSIVE"
    FAST_SLA_EXCEEDED = "FAST_SLA_EXCEEDED"
    UNREADABLE_DOCUMENT = "UNREADABLE_DOCUMENT"
    CONVERSION_FAILED = "CONVERSION_FAILED"
    LANGUAGE_UNAVAILABLE = "LANGUAGE_UNAVAILABLE"
    SERVICE_BUSY = "SERVICE_BUSY"
    NOT_READY = "NOT_READY"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    INVALID_INPUT = "INVALID_INPUT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    SLA_REJECTED = "SLA_REJECTED"
    RETRY_LIMIT_EXCEEDED = "RETRY_LIMIT_EXCEEDED"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    WORKSPACE_ERROR = "WORKSPACE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class EngineError(Exception):
    """A public error with an English message and explicitly safe scalar details."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        stage: str | None = None,
        details: Mapping[str, JSONScalar] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.stage = stage
        safe_details = dict(details or {})
        for key, value in safe_details.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("Error details must contain string keys and JSON scalar values.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Error details must contain finite numbers.")
        self.details = MappingProxyType(safe_details)

    def to_dict(self, *, request_id: str | None = None) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "stage": self.stage,
            "request_id": request_id,
            "details": dict(self.details),
        }


def public_error(error: BaseException) -> EngineError:
    """Preserve structured errors; redact unknown exception messages at the boundary."""
    if isinstance(error, EngineError):
        return error
    return EngineError(ErrorCode.INTERNAL_ERROR, "An unexpected internal error occurred.")
