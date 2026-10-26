import pandas as pd
import pytest

from insurance_analytics.targets import add_claim_targets


def test_missing_claim_amount_is_unknown_by_default():
    source = pd.DataFrame({"TotalClaims": [None, 0.0, 125.5]})
    result = add_claim_targets(source)

    assert pd.isna(result["HadClaim"].iloc[0])
    assert result["HadClaim"].iloc[1] == 0
    assert result["HadClaim"].iloc[2] == 1
    assert result["ClaimStatus"].tolist() == ["unknown", "no_claim", "claim"]
    assert pd.isna(result["ClaimSeverityTarget"].iloc[0])
    assert pd.isna(result["ClaimSeverityTarget"].iloc[1])
    assert result["ClaimSeverityTarget"].iloc[2] == 125.5


def test_missing_can_be_no_claim_only_when_explicitly_configured():
    source = pd.DataFrame({"TotalClaims": [None, 100.0]})
    result = add_claim_targets(source, missing_claims="no_claim")

    assert result["HadClaim"].iloc[0] == 0
    assert result["HadClaim"].iloc[1] == 1
    assert pd.isna(result["ClaimSeverityTarget"].iloc[0])


def test_negative_claim_amount_is_flagged_not_absed():
    source = pd.DataFrame({"TotalClaims": [-10.0]})
    result = add_claim_targets(source)

    assert result["ClaimStatus"].iloc[0] == "invalid"
    assert pd.isna(result["HadClaim"].iloc[0])
    assert pd.isna(result["ClaimSeverityTarget"].iloc[0])
    assert result["ClaimAmountObserved"].iloc[0] == -10.0


def test_input_is_not_mutated():
    source = pd.DataFrame({"TotalClaims": [None, 12.0]})
    original = source.copy(deep=True)
    add_claim_targets(source)
    pd.testing.assert_frame_equal(source, original)


def test_missing_required_column_raises():
    with pytest.raises(KeyError, match="TotalClaims"):
        add_claim_targets(pd.DataFrame({"premium": [10.0]}))


def test_invalid_missing_policy_raises():
    with pytest.raises(ValueError, match="missing_claims"):
        add_claim_targets(pd.DataFrame({"TotalClaims": [None]}), missing_claims="guess")
