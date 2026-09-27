"""Best-effort process memory observations without a mandatory GPU dependency."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ByteReader = Callable[[], int | None]


def read_rss_bytes() -> int | None:
    """Read current process RSS on Linux; unsupported platforms report unavailable."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def read_process_high_water_bytes() -> int | None:
    """Return the process-lifetime RSS high-water mark, not a per-request peak."""
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return int(raw * 1024)
        if sys.platform == "darwin":
            return int(raw)
    except (ImportError, OSError, ValueError):
        pass
    return None


@dataclass
class _Observations:
    start: int | None = None
    end: int | None = None
    peak: int | None = None
    samples: int = 0
    missing_samples: int = 0

    def add(self, value: int | None) -> None:
        if self.samples + self.missing_samples == 0:
            self.start = value
        self.end = value
        if value is None:
            self.missing_samples += 1
        else:
            self.samples += 1
            self.peak = max(value, self.peak if self.peak is not None else value)


class MemoryTracker:
    """Sample at lifecycle/stage boundaries; brief allocation peaks can be missed.

    VRAM must be supplied by a local backend as process-scoped bytes. Device-wide
    usage is not a request measurement and must not be passed to this tracker.
    """

    def __init__(
        self,
        *,
        rss_reader: ByteReader = read_rss_bytes,
        vram_reader: ByteReader | None = None,
    ) -> None:
        self._rss_reader = rss_reader
        self._vram_reader = vram_reader
        self._ram = _Observations()
        self._vram = _Observations()

    @staticmethod
    def _safe_read(reader: ByteReader | None) -> int | None:
        if reader is None:
            return None
        try:
            value = reader()
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            return value
        except Exception:
            # Optional telemetry must not turn an otherwise valid request into an error.
            return None

    def sample(self) -> None:
        self._ram.add(self._safe_read(self._rss_reader))
        self._vram.add(self._safe_read(self._vram_reader))

    def snapshot(self) -> dict[str, object]:
        return {
            "ram_start_bytes": self._ram.start,
            "ram_end_bytes": self._ram.end,
            "ram_peak_bytes": self._ram.peak,
            "ram_samples": self._ram.samples,
            "ram_missing_samples": self._ram.missing_samples,
            "ram_process_high_water_bytes": read_process_high_water_bytes(),
            "vram_start_bytes": self._vram.start,
            "vram_end_bytes": self._vram.end,
            "vram_peak_bytes": self._vram.peak,
            "vram_samples": self._vram.samples,
            "vram_missing_samples": self._vram.missing_samples,
            "vram_provider_configured": self._vram_reader is not None,
            "memory_scope": "process",
            "memory_sampling": "request_and_stage_boundaries",
        }
