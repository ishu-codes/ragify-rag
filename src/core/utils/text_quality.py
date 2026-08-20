"""Heuristics for detecting degenerate extracted text / chunks.

Catches garbage produced by broken PDF extraction (e.g. one character per
line) before it is embedded, indexed, or used as retrieval context, so the
pipeline degrades explicitly instead of generating confident answers from
meaningless context.
"""

MIN_CONTENT_CHARS = 20
MIN_LINES_FOR_FRAGMENT_CHECK = 5
MAX_FRAGMENT_LINE_RATIO = 0.6
MIN_AVG_CHARS_PER_LINE = 3.0


def is_degenerate(text: str | None) -> bool:
    """Return True when the text looks like extraction garbage, not prose."""
    if not text:
        return True
    text = text.strip()
    if len(text) < MIN_CONTENT_CHARS:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < MIN_LINES_FOR_FRAGMENT_CHECK:
        return False

    fragment_lines = sum(1 for line in lines if len(line) <= 2)
    if fragment_lines / len(lines) >= MAX_FRAGMENT_LINE_RATIO:
        return True

    avg_chars_per_line = sum(len(line) for line in lines) / len(lines)
    return avg_chars_per_line < MIN_AVG_CHARS_PER_LINE
