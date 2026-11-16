"""Credential masking and the shared outbound-text boundary.

The masking rule is t6's ``t6.privacy.withhold``. This module keeps the names the
call sites use: ``outbound_text``, ``mask_text``, ``mask_result``, ``mask_blocks``
and ``MaskedText``.

V1 deliberately does not guess: unlabeled credentials and labels outside the
supported set are known false negatives. URL tokens are preserved verbatim,
including embedded access keys, because changing them would break source links.
"""

from __future__ import annotations

from typing import NamedTuple

from t6.privacy.withhold import MASKER_VERSION, WITHHELD, withhold

__all__ = [
    "MASKER_VERSION",
    "WITHHELD",
    "MaskedText",
    "mask_blocks",
    "mask_result",
    "mask_text",
    "outbound_text",
]


class MaskedText(NamedTuple):
    """Text approved to leave the protected ledger, with replay metadata."""

    text: str
    withheld_count: int
    masker_version: str


def outbound_text(text: str) -> MaskedText:
    """Return the one safe representation for outbound source-derived text."""
    result = withhold(text)
    return MaskedText(result.text, result.count, result.masker_version)


def mask_result(text: str) -> tuple[str, int]:
    """Return masked text and the number of newly withheld assignments."""
    result = outbound_text(text)
    return result.text, result.withheld_count


def mask_text(text: str) -> str:
    """Return the outbound-safe representation of ``text``."""
    return outbound_text(text).text


def mask_blocks(blocks: list[str]) -> list[str]:
    """Apply :func:`mask_text` to each block."""
    return [mask_text(b) for b in blocks]
