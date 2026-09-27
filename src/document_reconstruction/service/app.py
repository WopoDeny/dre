"""A small WSGI API. Production HTTP handling belongs to Gunicorn."""

from __future__ import annotations

import hmac
import json
import sys
from datetime import datetime, timezone
from uuid import uuid4

from ..core import DeadlineBudget, EngineError, ErrorCode
from ..core.errors import public_error
from ..engine import MAX_UPLOAD_BYTES

ACCEPTED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class Application:
    def __init__(self, pool: object, *, api_key: str) -> None:
        self.pool = pool
        self.api_key = api_key

    def __call__(self, environ: dict[str, object], start_response: object) -> list[bytes]:
        budget = DeadlineBudget()
        request_id = uuid4().hex

        def respond(status: str, body: object, media_type: str = "application/json", extra: list[tuple[str, str]] | None = None) -> list[bytes]:
            encoded = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
            headers = [("Content-Type", media_type), ("Content-Length", str(len(encoded))), ("X-Request-ID", request_id), ("Cache-Control", "no-store")]
            start_response(status, headers + (extra or []))
            if environ.get("PATH_INFO") == "/convert":
                _log(request_id, status, body if isinstance(body, dict) else None, budget.elapsed_ms, len(encoded))
            return [encoded]

        try:
            path, method = environ.get("PATH_INFO", "/"), environ.get("REQUEST_METHOD", "GET")
            if path == "/health" and method == "GET":
                return respond("200 OK", {"status": "alive"})
            if path == "/ready" and method == "GET":
                ready = self.pool.ready and bool(self.api_key)
                return respond("200 OK" if ready else "503 Service Unavailable", {"ready": ready, "installed_ocr_languages": self.pool.languages, "ocr_capabilities": getattr(self.pool, "ocr_capabilities", {})})
            if path != "/convert":
                return respond("404 Not Found", {"error": {"code": "NOT_FOUND", "message": "The endpoint does not exist.", "request_id": request_id}})
            if method != "POST":
                return respond("405 Method Not Allowed", {"error": {"code": "METHOD_NOT_ALLOWED", "message": "Use POST for conversion.", "request_id": request_id}}, extra=[("Allow", "POST")])
            if not self.api_key:
                raise EngineError(ErrorCode.NOT_READY, "The service API key has not been configured.")
            authorization = str(environ.get("HTTP_AUTHORIZATION", ""))
            if not hmac.compare_digest(authorization, "Bearer " + self.api_key):
                return respond("401 Unauthorized", {"error": {"code": "UNAUTHORIZED", "message": "A valid bearer token is required.", "request_id": request_id}})
            if environ.get("CONTENT_TYPE", "").split(";")[0].lower() not in ACCEPTED_TYPES:
                return respond("415 Unsupported Media Type", {"error": {"code": "INVALID_PDF", "message": "Send a PDF or an image (JPEG, PNG, WEBP, HEIC) as the request body.", "request_id": request_id}})
            try:
                length = int(environ.get("CONTENT_LENGTH", ""))
            except (ValueError, TypeError):
                raise EngineError(ErrorCode.INVALID_PDF, "A valid Content-Length is required.") from None
            if length > MAX_UPLOAD_BYTES:
                raise EngineError(ErrorCode.UPLOAD_TOO_LARGE, "The PDF exceeds the 20 MiB upload limit.")
            if length <= 0:
                raise EngineError(ErrorCode.INVALID_PDF, "The PDF request body is empty.")
            chunks = []
            remaining = length
            while remaining:
                budget.check(stage="upload")
                chunk = environ["wsgi.input"].read(min(65_536, remaining))
                if not chunk:
                    raise EngineError(ErrorCode.INVALID_PDF, "The PDF request body is incomplete.")
                chunks.append(chunk)
                remaining -= len(chunk)
            result = self.pool.convert(b"".join(chunks), budget=budget, request_id=request_id)
            if result.error:
                error = result.error
                return respond(_status(error["code"]), {"error": error, "metrics": result.metrics})
            # The pool may have synchronized before a slow final handoff. Do
            # not emit HTTP 200 for a result whose server budget has expired.
            budget.check(stage="response")
            return respond("200 OK", result.docx, DOCX_MEDIA_TYPE, [("Content-Disposition", f'attachment; filename="{request_id}.docx"'), ("X-Conversion-MS", str(round(budget.elapsed_ms, 2)))])
        except Exception as cause:
            error = public_error(cause)
            if error.code == ErrorCode.DEADLINE_EXCEEDED:
                error = EngineError(ErrorCode.FAST_SLA_EXCEEDED, "The request deadline has been reached.", stage=error.stage)
            return respond(_status(error.code), {"error": error.to_dict(request_id=request_id)})


def _log(request_id: str, status: str, body: dict | None, elapsed_ms: float, size: int) -> None:
    """One JSON line per conversion on stderr (Gunicorn's error log, `docker logs`): outcome only, no document content."""
    error = (body or {}).get("error") or {}
    details = error.get("details") or {}
    metrics = (body or {}).get("metrics") or {}
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": request_id, "status": int(status.split()[0]),
              "ms": round(elapsed_ms), "code": error.get("code"), "reason": details.get("reason"), "stage": error.get("stage"),
              "page": details.get("page_index"), "pages": metrics.get("pages_total"), "docx_bytes": size if not error else None}
    print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)


def _status(code: str) -> str:
    return {
        "INVALID_PDF": "400 Bad Request",
        "UPLOAD_TOO_LARGE": "413 Payload Too Large",
        "DOCUMENT_TOO_EXPENSIVE": "422 Unprocessable Content",
        "LANGUAGE_UNAVAILABLE": "422 Unprocessable Content",
        "UNREADABLE_DOCUMENT": "422 Unprocessable Content",
        "FAST_SLA_EXCEEDED": "504 Gateway Timeout",
        "SERVICE_BUSY": "503 Service Unavailable",
        "NOT_READY": "503 Service Unavailable",
    }.get(code, "500 Internal Server Error")
