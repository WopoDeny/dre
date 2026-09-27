"""Shared infrastructure with no recognition or DOCX dependency."""

from .context import ConversionContext, PageRoute
from .deadline import DeadlineBudget
from .errors import EngineError, ErrorCode
from .metrics import Metrics, Stage, StageTimer

__all__ = ["ConversionContext", "DeadlineBudget", "EngineError", "ErrorCode", "Metrics", "PageRoute", "Stage", "StageTimer"]
