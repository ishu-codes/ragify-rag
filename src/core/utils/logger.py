"""Structured JSON logging for the rag service.

Every record is emitted as a single-line JSON object with a stable schema:

    {"ts", "level", "logger", "event", "correlation_id", ...extra fields}

The correlation id comes from the :data:`correlation_id` context variable, so
a request's stages can be traced end to end without threading the id through
every call site.  Third-party HTTP clients (httpx, qdrant, ...) are kept at
WARNING so INFO logs stay meaningful.
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "urllib3", "openai", "qdrant_client")

# Attributes every LogRecord carries; anything else logged via `extra` is
# treated as structured payload.
_STD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
)

_ANSI = {
    "red": "\033[31m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}


def _paint(text: str, color: str) -> str:
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        request_id = correlation_id.get()
        if request_id:
            payload["correlation_id"] = request_id
        for key, value in record.__dict__.items():
            if key in _STD_RECORD_ATTRS or key in {"message", "asctime"}:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable, colorized output for interactive terminals.

    Pipeline events are rendered with the classic colors: stage in red,
    context in cyan, result in green, timings in yellow.  Colors are ANSI
    codes and only make sense in a terminal; non-event records fall back to a
    plain ``[LEVEL] event {json extras}`` line.
    """

    def format(self, record: logging.LogRecord) -> str:
        event = record.getMessage()
        fields = record.__dict__

        if event == "step_start":
            lines = [_paint(str(fields.get("title", "")), "red")]
            details = fields.get("details")
            if details:
                lines.append(_paint(str(details), "cyan"))
            return "\n".join(lines)

        if event == "step_end":
            lines = []
            result = fields.get("result")
            if result:
                lines.append(_paint(str(result), "green"))
            lines.append(
                _paint(
                    f"Took {float(fields.get('duration_s', 0)):.2f}s since previous step",
                    "yellow",
                )
            )
            return "\n".join(lines)

        if event == "pipeline_end":
            return _paint(
                f"Total time taken: {float(fields.get('total_s', 0)):.2f}s",
                "yellow",
            )

        if event == "step_error":
            lines = [_paint(f"Error in {fields.get('step', '')}", "red")]
            error = fields.get("error")
            if error:
                lines.append(_paint(str(error), "red"))
            return "\n".join(lines)

        extras = {
            key: value
            for key, value in fields.items()
            if key not in _STD_RECORD_ATTRS and key not in {"message", "asctime"}
        }
        suffix = f" {json.dumps(extras, default=str)}" if extras else ""
        return f"[{record.levelname}] {event}{suffix}"


def get_logger(name: str = "ragify") -> logging.Logger:
    """Return a logger, configuring the root handler once on first use."""
    root = logging.getLogger()
    if not root.handlers:
        # RAGIFY_LOG_FORMAT: auto | json | console.  Auto emits JSON when
        # stdout is not a terminal (pipes, containers) and colorized console
        # output when it is.
        fmt = os.getenv("RAGIFY_LOG_FORMAT", "auto")
        use_json = fmt == "json" or (fmt == "auto" and not sys.stdout.isatty())
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter() if use_json else ConsoleFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        for noisy in _NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)
