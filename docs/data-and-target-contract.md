# Data and Target Contract

## Scope

This repository currently describes the AlphaCare Insurance Solutions (ACIS)
South African motor-insurance dataset. Its definitions and results must not be
assumed to transfer directly to Awash Insurance or another market.

## Claim outcomes

The raw source field TotalClaims is an outcome field, not an ordinary
predictor. Never median-impute it before defining claim occurrence or selecting
severity observations.

The reusable helper insurance_analytics.targets.add_claim_targets applies
these conservative defaults:

| Raw amount | Occurrence target | Severity target | Status |
|---|---:|---:|---|
| Positive observed amount | 1 | Observed amount | claim |
| Observed zero | 0 | Missing / excluded | no_claim |
| Missing amount | Unknown (<NA>) | Missing / excluded | unknown |
| Negative amount | Unknown (<NA>) | Missing / excluded | invalid |

Missing amounts may represent no-claim records in some source datasets, but
that is a data-contract decision, not a general statistical truth. Use
missing_claims='no_claim' only after confirming the convention from the
authoritative data dictionary or data owner. Record that decision in the
analysis configuration and report.

Negative values must be investigated against the source definitions and
accounting conventions. Do not use abs() to silently erase their sign.

## Model target definitions

- Claim occurrence: a verified binary outcome. Unknown outcomes are excluded
  from supervised training and reported in data-quality summaries.
- Claim severity: positive, observed claim amounts only. No-claim, unknown,
  and invalid rows are excluded from severity-target training.
- Expected claims cost: combine frequency and conditional severity estimates
  only when observation units, exposure periods, currency, and claim
  definitions are aligned.
- Historical premium prediction: predicts recorded premium, not actuarially
  adequate or technically indicated premium. Do not label it as a pricing
  recommendation without a separate actuarial methodology.

## Leakage and validation

1. Split data before fitting imputers, encoders, scalers, feature selectors,
   resampling steps, or other learned transformations.
2. Fit transformations only on training data, preferably inside a scikit-learn
   Pipeline / ColumnTransformer.
3. Do not use target-derived fields such as HadClaim, LossRatio, or TotalClaims
   as predictors for claim occurrence when they reveal the outcome or would not
   be available at scoring time.
4. Prefer a time-aware holdout when transaction dates and intended deployment
   make future prediction the use case. Prevent policy/customer leakage where
   repeated records share the same underlying risk.
5. Report class prevalence and a baseline. For rare claims, accuracy alone is
   insufficient; report precision, recall, PR-AUC, confusion matrix at a
   declared threshold, and calibration where appropriate.
6. Distinguish row-level records from policies, insured exposures, and claims.
   Confirm the unit of analysis before aggregating or splitting.

## Data handling

- Do not commit raw policyholder-level data, credentials, or sensitive exports.
- Keep large/private datasets outside Git and document approved retrieval.
- Preserve raw fields and report validation issues instead of silently changing
  financial values.
- Treat model outputs as analytical evidence, not production decisions, until
  independently validated for the target market and use case.
