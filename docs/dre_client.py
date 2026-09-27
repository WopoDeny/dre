"""Minimal client for the Document Reconstruction Engine HTTP API.

Usage:
    from dre_client import convert_document
    docx_bytes = convert_document("/path/to/file.pdf")  # bytes or None

Configuration via environment variables:
    DRE_URL      e.g. http://10.0.0.5:8080/convert
    DRE_API_KEY  the key the service was started with
    DRE_TIMEOUT  seconds to wait for one document (default 120)
"""

import logging
import os
from pathlib import Path

import requests

log = logging.getLogger("dre_client")

DRE_URL = os.environ.get("DRE_URL", "http://localhost:8080/convert")
DRE_API_KEY = os.environ.get("DRE_API_KEY", "")
DRE_TIMEOUT = float(os.environ.get("DRE_TIMEOUT", "120"))


def convert_document(path):
    """Send one PDF/JPG/PNG/WEBP file and return DOCX bytes, or None on refusal or error."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        log.error("DRE: cannot read %s: %s", path, exc)
        return None

    try:
        resp = requests.post(
            DRE_URL,
            data=data,
            headers={"Authorization": f"Bearer {DRE_API_KEY}"},
            timeout=DRE_TIMEOUT,
        )
    except requests.RequestException as exc:
        log.error("DRE: request failed for %s: %s", path.name, exc)
        return None

    # A DOCX file is a ZIP archive and always starts with "PK".
    if resp.status_code == 200 and resp.content[:2] == b"PK":
        return resp.content

    reason = resp.text[:300].replace("\n", " ")
    log.warning("DRE: %s not converted (HTTP %s): %s", path.name, resp.status_code, reason)
    return None
