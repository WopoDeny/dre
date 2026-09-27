"""Send every document in a folder to the DRE server, one by one.

Setup (once):
    pip install requests

Run:
    DRE_URL=http://SERVER_IP:8080/convert DRE_API_KEY=KEY python3 test_remote.py IN_DIR OUT_DIR

Results:
    OUT_DIR/ok/       converted DOCX files
    OUT_DIR/refused/  copies of refused files plus <name>.reason.txt
    OUT_DIR/log.csv   file, status, seconds, HTTP code, reason
"""

import csv
import os
import shutil
import sys
import time
from pathlib import Path

import requests

URL = os.environ.get("DRE_URL", "http://localhost:8080/convert")
KEY = os.environ.get("DRE_API_KEY", "")
TIMEOUT = float(os.environ.get("DRE_TIMEOUT", "120"))
EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ok_dir, bad_dir = dst / "ok", dst / "refused"
    ok_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    ok = bad = 0
    with open(dst / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "seconds", "http", "reason"])
        for p in files:
            t0 = time.time()
            code, reason = "", ""
            try:
                r = requests.post(URL, data=p.read_bytes(),
                                  headers={"Authorization": f"Bearer {KEY}"}, timeout=TIMEOUT)
                code = r.status_code
                if code == 200 and r.content[:2] == b"PK":
                    (ok_dir / (p.stem + ".docx")).write_bytes(r.content)
                    status = "ok"
                else:
                    status, reason = "refused", r.text[:300].replace("\n", " ")
            except requests.RequestException as exc:
                status, reason = "error", str(exc)
            dt = round(time.time() - t0, 1)
            if status == "ok":
                ok += 1
            else:
                bad += 1
                shutil.copy2(p, bad_dir / p.name)
                (bad_dir / (p.name + ".reason.txt")).write_text(reason, encoding="utf-8")
            w.writerow([p.name, status, dt, code, reason])
            print(f"{status:8} {dt:6.1f}s  {p.name}  {reason[:80]}")

    print(f"\nTotal: {len(files)}, converted: {ok}, refused/errors: {bad}")


if __name__ == "__main__":
    main()
