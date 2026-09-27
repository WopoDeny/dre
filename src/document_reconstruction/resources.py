"""What the machine gives this process: cores and memory, container limits included.

`docker run --cpus=1 --memory=2g` is seen through cgroups, not through the host's
core count, so a container never plans more work than it may run.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# One engine process (OCR models, word lists, Hunspell) takes about this much memory,
# every further page thread about this much more (its own Tesseract models and images).
PROCESS_MB = 1800
PAGE_THREAD_MB = 400
MAX_PAGE_THREADS = 4


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_count() -> int:
    """Cores this process may use: affinity (taskset) and the cgroup quota (--cpus)."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    quota = None
    raw = _read("/sys/fs/cgroup/cpu.max")  # cgroup v2: "max 100000" or "100000 100000"
    if raw and not raw.startswith("max"):
        limit, period = raw.split()[:2]
        quota = int(limit) / int(period)
    else:
        limit, period = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if limit and period and int(limit) > 0:
            quota = int(limit) / int(period)
    if quota:
        cores = min(cores, max(1, math.floor(quota + 1e-6)))
    return max(1, cores)


def memory_mb() -> int | None:
    """Memory this process may use: the container limit or the available system memory."""
    limits = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read(path)
        if raw and raw.isdigit() and int(raw) < 1 << 60:
            limits.append(int(raw) // (1024 * 1024))
    meminfo = _read("/proc/meminfo") or ""
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            limits.append(int(line.split()[1]) // 1024)
    return min(limits) if limits else None


def page_threads(requested: str | None = None) -> int:
    """Pages recognized at once: DRE_PAGE_THREADS, or as many as cores and memory allow."""
    requested = (requested if requested is not None else os.environ.get("DRE_PAGE_THREADS", "auto")).strip().lower()
    if requested and requested != "auto":
        return max(1, min(MAX_PAGE_THREADS, int(requested)))
    threads = min(cpu_count(), MAX_PAGE_THREADS)
    memory = memory_mb()
    if memory is not None:
        threads = min(threads, max(1, 1 + (memory - PROCESS_MB) // PAGE_THREAD_MB))
    return max(1, threads)


def worker_rss_mb(threads: int | None = None) -> int:
    """Memory limit of one conversion worker process: DRE_WORKER_RSS_MB or the measured need."""
    fixed = os.environ.get("DRE_WORKER_RSS_MB", "").strip()
    if fixed:
        return int(fixed)
    threads = page_threads() if threads is None else threads
    need = PROCESS_MB + 400 + PAGE_THREAD_MB * (threads - 1)
    memory = memory_mb()
    if memory is not None and memory >= 16_000:
        return max(need, 8000)  # plenty of memory: the limit only stops a runaway worker
    return need


def workers_and_concurrency() -> tuple[int, int]:
    """(worker processes, documents at a time): DRE_WORKERS / DRE_CONCURRENCY, or by the machine.

    Documents at a time = cores (one core: one document). With memory to spare, one more warm worker waits
    so that a worker stopped at a deadline is replaced at once.
    """
    concurrency = int(os.environ.get("DRE_CONCURRENCY", "0") or 0) or cpu_count()
    fixed = os.environ.get("DRE_WORKERS", "auto").strip().lower()
    if fixed and fixed != "auto":
        return max(1, int(fixed)), concurrency
    memory = memory_mb()
    spare = 1 if memory is not None and memory >= (concurrency + 1) * (worker_rss_mb() + 500) else 0
    return min(8, concurrency + spare), concurrency
