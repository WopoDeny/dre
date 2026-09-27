"""Gunicorn entry point; initialize one bounded warm pool per service process."""

import atexit
import os

from .app import Application
from ..resources import worker_rss_mb, workers_and_concurrency
from .workers import WarmWorkerPool

workers, concurrency = workers_and_concurrency()
pool = WarmWorkerPool(workers=workers, concurrency=concurrency, max_worker_rss_mb=worker_rss_mb())
atexit.register(pool.close)
application = Application(pool, api_key=os.environ.get("DRE_API_KEY", ""))
