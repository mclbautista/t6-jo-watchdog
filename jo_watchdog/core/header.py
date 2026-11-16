"""Pure core: Job Order header parsing, canonical Name, and JO Number.

The rules are t6's (``t6.domain.job_order_rules``). This module keeps the import
path the call sites use. Callers pass ``config.department_aliases``; they do not
rely on t6's default alias map.
"""

from __future__ import annotations

from t6.domain.job_order_rules import (
    Header,
    canonical_name,
    normalize_department,
    parse_header,
    parse_number,
)

__all__ = ["Header", "canonical_name", "normalize_department", "parse_header", "parse_number"]
