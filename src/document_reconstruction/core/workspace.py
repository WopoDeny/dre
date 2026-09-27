"""Owned temporary directories with explicit cleanup and confined artifact paths."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .errors import EngineError, ErrorCode


class TemporaryWorkspace:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._closed = False
        self.cleanup_error: EngineError | None = None

    def __enter__(self) -> TemporaryWorkspace:
        if self._directory is not None or self._closed:
            raise RuntimeError("A temporary workspace is single-use.")
        try:
            self._directory = tempfile.TemporaryDirectory(prefix="dre-", dir=self._base_dir)
        except OSError as error:
            raise EngineError(ErrorCode.WORKSPACE_ERROR, "The temporary workspace could not be created.") from error
        return self

    @property
    def path(self) -> Path:
        if self._directory is None or self._closed:
            raise RuntimeError("The temporary workspace is not active.")
        return Path(self._directory.name).resolve()

    def file(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        root = self.path
        target = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or target == root or not target.is_relative_to(root):
            raise EngineError(ErrorCode.WORKSPACE_ERROR, "The workspace path must remain inside the request directory.")
        return target

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            if self._directory is not None:
                self._directory.cleanup()
        except OSError as error:
            self.cleanup_error = EngineError(ErrorCode.WORKSPACE_ERROR, "The temporary workspace could not be removed.")
            raise self.cleanup_error from error
        self._closed = True

    def __exit__(self, exc_type: object, error: BaseException | None, traceback: object) -> bool:
        try:
            self.cleanup()
        except EngineError:
            if error is None:
                raise
            error.add_note("Temporary workspace cleanup also failed.")
        return False
