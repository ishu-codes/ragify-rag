"""Step event logging for the RAG pipeline.

Every pipeline step prints:
  - the event title in red, followed by its context in cyan (before running),
  - the result in green and the time it took since the previous step in
    yellow (after it finishes),
  - five blank lines to separate events.
"""

import time

from src.utils.colors import colorize

_last_step_start: float | None = None
_pipeline_start: float | None = None


def start_event(title: str, context: str = "") -> None:
    """Announce a pipeline step right before it runs."""
    global _last_step_start, _pipeline_start
    _last_step_start = time.monotonic()
    if _pipeline_start is None:
        _pipeline_start = _last_step_start

    print(colorize(f"\n{title}", "RED"), flush=True)
    if context:
        print(colorize(context, "CYAN"), flush=True)


def end_event(result: str = "") -> None:
    """Report the step result and its duration since the previous step."""
    global _last_step_start

    elapsed = 0.0
    if _last_step_start is not None:
        elapsed = time.monotonic() - _last_step_start

    if result:
        print(colorize(result, "GREEN"), flush=True)
    print(colorize(f"Took {elapsed:.2f}s since previous step", "YELLOW"), flush=True)
    print("\n" * 5, end="", flush=True)


def end_pipeline() -> None:
    """Print the total time taken by the whole pipeline and reset the timer."""
    global _pipeline_start
    if _pipeline_start is not None:
        total = time.monotonic() - _pipeline_start
        print(colorize(f"Total time taken: {total:.2f}s", "YELLOW"), flush=True)
        print("\n" * 5, end="", flush=True)
        _pipeline_start = None


def error_event(error: str, context: str = "") -> None:
    """Print the error detail for a failed step (in red)."""
    if context:
        print(colorize(f"\nError in {context}", "RED"), flush=True)
    else:
        print(colorize("\nError", "RED"), flush=True)
    print(colorize(error, "RED"), flush=True)
    print("\n" * 5, end="", flush=True)
