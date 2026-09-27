"""python -m document_reconstruction  FILE --output OUT.docx | selfcheck | batch IN_DIR OUT_DIR | summary OUT_DIR"""

import sys


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "selfcheck":
        from .selfcheck import main as selfcheck
        return selfcheck(sys.argv[2:])
    if command == "summary":
        from .batch import summary
        return summary(sys.argv[2:])
    if command == "batch":
        from .batch import main as batch
        return batch(sys.argv[2:])
    from .cli import main as convert
    return convert()


if __name__ == "__main__":
    raise SystemExit(main())
