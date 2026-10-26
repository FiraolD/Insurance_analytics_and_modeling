"""Explicit, leakage-resistant construction of insurance claim targets.

Important: missing claim amounts are not automatically interpreted as zero.
Set missing_claims='no_claim' only after verifying that the source data
 dictionary defines a missing TotalClaims value as an observed no-claim outcome.
Negative amounts are flagged for review rather than converted to absolute values.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

MissingClaimsPolicy = Literal["unknown", "no_claim"]


def add_claim_targets(
    frame: pd.DataFrame,
    claims_col: str = "TotalClaims",
    missing_claims: MissingClaimsPolicy = "unknown",
) -> pd.DataFrame:
    """Return a copy with explicit occurrence and severity target columns.

    Adds ClaimAmountObserved, ClaimStatus, HadClaim (nullable integer), and
    ClaimSeverityTarget (positive observed claim amounts only). All other rows
    have a missing severity target and must not enter severity training.

    Recorded zero means no claim; a positive amount means a claim. Missing
    values follow the explicit policy. Negative amounts are invalid until their
    accounting meaning is investigated. The input DataFrame is never modified.
    """
    if claims_col not in frame.columns:
        raise KeyError(f"Required claims column {claims_col!r} is missing.")
    if missing_claims not in ("unknown", "no_claim"):
        raise ValueError("missing_claims must be either 'unknown' or 'no_claim'.")

    result = frame.copy()
    amount = pd.to_numeric(result[claims_col], errors="coerce")
    result["ClaimAmountObserved"] = amount

    status = pd.Series("unknown", index=result.index, dtype="string")
    status.loc[amount.notna() & amount.eq(0)] = "no_claim"
    status.loc[amount.notna() & amount.gt(0)] = "claim"
    status.loc[amount.notna() & amount.lt(0)] = "invalid"

    if missing_claims == "no_claim":
        status.loc[amount.isna()] = "no_claim"

    occurrence = pd.Series(pd.NA, index=result.index, dtype="Int64")
    occurrence.loc[status.eq("no_claim")] = 0
    occurrence.loc[status.eq("claim")] = 1

    severity = amount.where(status.eq("claim"))

    result["ClaimStatus"] = status
    result["HadClaim"] = occurrence
    result["ClaimSeverityTarget"] = severity
    return result
