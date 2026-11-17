"""Error text that holds no message text: a model's output or a posted message can sit in a
validation error's input values, so an error stored or alerted keeps the class, the count and
the field paths only."""

from __future__ import annotations

from pydantic import ValidationError


def describe_validation_error(exc: ValidationError) -> str:
    paths = sorted({".".join(str(p) for p in e["loc"]) or "(root)" for e in exc.errors()})
    return f"{type(exc).__name__}: {exc.error_count()} errors at {', '.join(paths)}"[:500]


def describe(exc: BaseException) -> str:
    """``Class: detail`` for an error whose detail is safe to store; class and paths otherwise."""
    if isinstance(exc, ValidationError):
        return describe_validation_error(exc)
    return f"{type(exc).__name__}: {exc}"
