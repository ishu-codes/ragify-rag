"""Step event logging for the RAG pipeline.

Every pipeline step is emitted as a structured JSON log record with stage
timing, so the output is greppable and ships to log aggregators unchanged.
"""

import time

from src.core.utils.logger import get_logger

logger = get_logger("ragify.events")

_last_step_start: float | None = None
_pipeline_start: float | None = None

_MAX_DETAILS = 2000
_MAX_RESULT = 500


def start_event(title: str, context: str = "") -> None:
    """Announce a pipeline step right before it runs."""
    global _last_step_start, _pipeline_start
    _last_step_start = time.monotonic()
    if _pipeline_start is None:
        _pipeline_start = _last_step_start
    logger.info(
        "step_start",
        extra={
            "title": title,
            "details": context[:_MAX_DETAILS],
        },
    )


def end_event(result: str = "") -> None:
    """Report the step result and its duration since the previous step."""
    global _last_step_start
    elapsed = 0.0
    if _last_step_start is not None:
        elapsed = time.monotonic() - _last_step_start
    logger.info(
        "step_end",
        extra={
            "duration_s": round(elapsed, 3),
            "result": result[:_MAX_RESULT],
        },
    )


def end_pipeline() -> None:
    """Log the total pipeline time and reset the timer."""
    global _pipeline_start
    if _pipeline_start is not None:
        total = time.monotonic() - _pipeline_start
        logger.info("pipeline_end", extra={"total_s": round(total, 3)})
        _pipeline_start = None


def error_event(error: str, context: str = "") -> None:
    """Log the error detail for a failed step."""
    logger.error(
        "step_error",
        extra={
            "step": context,
            "error": error[:_MAX_RESULT],
        },
    )
