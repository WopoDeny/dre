import os

# Legacy native-PDF regressions cover Arabic and Chinese text layers. Stage 1
# disables those scripts by default (DRE_NATIVE_ALLOWED_SCRIPTS); the tests keep
# them enabled so the native Arabic/CJK path cannot regress unnoticed.
os.environ.setdefault("DRE_NATIVE_ALLOWED_SCRIPTS", "Latin,Cyrillic,Arabic,Han")
