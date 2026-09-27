"""Convert one local PDF with the same supervised engine used by the service."""

import argparse
import json
from pathlib import Path

from .core.errors import public_error
from .resources import worker_rss_mb
from .service.workers import WarmWorkerPool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("The output already exists. Choose a new output path.")
    pool = None
    try:
        payload = args.input.read_bytes()
        pool = WarmWorkerPool(max_worker_rss_mb=worker_rss_mb())
        result = pool.convert(payload)
        if not result.ok:
            print(json.dumps({"error": result.error, "metrics": result.metrics}))
            return 1
        with args.output.open("xb") as output:
            output.write(result.docx)
        print(json.dumps({"request_id": result.request_id, "output": str(args.output), "metrics": result.metrics}))
        return 0
    except Exception as error:
        print(json.dumps({"error": public_error(error).to_dict()}))
        return 1
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
