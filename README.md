# Insurance Risk Analytics & Predictive Modelling

A Python project for exploratory analysis, statistical testing, and predictive
modelling using historical motor-insurance data associated with AlphaCare
Insurance Solutions (ACIS), South Africa.

> **Status: reliability improvements in progress.** Existing scripts and
> previously reported model metrics are exploratory. They should not be
> interpreted as validated pricing recommendations until the target definitions,
> data assumptions, preprocessing, and evaluation results have been reproduced.

## Objectives

- Understand portfolio patterns across vehicle, policyholder, and geographic
  attributes.
- Evaluate statistical hypotheses with clearly stated assumptions.
- Develop separate models for claim occurrence, claim severity, and historical
  premium analysis.
- Produce reproducible reports that distinguish evidence from assumptions.
- Establish quality controls before using model results to support decisions.

## Dataset and responsible use

The existing project documentation describes historical South African motor
insurance data covering February 2014 through August 2015. Confirm the exact
data dictionary, observation unit, currency, exposure period, and permitted use
before drawing conclusions. The raw dataset is not distributed in this
repository; do not commit policyholder-level records or confidential exports.

The data and findings are not representative of Awash Insurance or the
Ethiopian market without independent validation on appropriate local data.

## Repository map

- `src/Modelling/`: existing modelling scripts, under reliability review.
- `src/EDA.py`: existing exploratory analysis script.
- `src/Hypothesis Tester.py`: existing hypothesis-testing script.
- `src/insurance_analytics/`: reusable components introduced during the
  reliability refactor.
- `tests/`: automated checks for core data and target logic.
- `docs/data-and-target-contract.md`: target semantics, leakage controls, and
  data-handling rules.
- `Reports/`: historical analysis artifacts; figures and metrics require
  reproduction before being treated as validated.

File names and run instructions in the original scripts are being reconciled
as part of the refactor. Do not assume every legacy script currently runs
end-to-end.

## Requirements

Use Python 3.10 or newer. Create and activate a virtual environment, then run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the existing model scripts and data-versioning workflow, install the
additional modelling dependencies:

```bash
python -m pip install -e ".[modelling,dev]"
```

The `requirements.txt` file is also provided for a conventional dependency
installation. Dependency ranges are declared, but a fully resolved lockfile
has not yet been generated.

## Run quality checks

```bash
python -m pytest
```

The initial tests cover claim-target construction. They do not yet certify that
all legacy analysis scripts or the full modelling workflow execute correctly.

## Critical target-definition rules

- A positive observed `TotalClaims` amount is treated as a claim.
- An observed zero is treated as no claim.
- A missing amount is **unknown by default**, not automatically zero.
- A negative amount is flagged for investigation rather than converted to an
  absolute value.
- Claim-severity training uses positive, observed claim amounts only.
- Target fields and target-derived metrics must not leak into model predictors.
- Fit imputers, encoders, scalers, feature selectors, and resampling steps only
  on training data, using a pipeline wherever possible.

These rules are implemented in
`insurance_analytics.targets.add_claim_targets`. The missing-value policy may
only be changed after verifying the authoritative source-data convention.
Details are documented in `docs/data-and-target-contract.md`.

## Modelling principles

1. Establish a transparent baseline before tuning complex algorithms.
2. Split data before fitting any learned preprocessing.
3. Use a time-aware holdout when the intended task is future-risk prediction.
4. For rare claim outcomes, report prevalence, precision, recall, PR-AUC,
   threshold-specific confusion matrices, and calibration where appropriate;
   accuracy alone is inadequate.
5. Evaluate severity only on the correctly defined claim population.
6. Historical premium prediction is not equivalent to actuarially adequate
   premium calculation.
7. Preserve negative financial values for investigation until their accounting
   meaning is understood; do not silently rewrite them.
8. Record dataset versions, configuration, random seeds, metrics, and known
   limitations with each reproducible experiment.

## Development workflow

Work on a feature branch. Keep changes small and testable, run quality checks,
and review the diff before merging. Raw data, secrets, and credentials must
remain outside version control. The main branch should only receive reviewed,
tested changes.

## Current work

The reliability refactor is addressing, in order:

1. Environment and repository consistency.
2. Data-quality checks and verified target definitions.
3. Leakage-safe, reusable model pipelines.
4. Baseline comparisons and appropriate evaluation metrics.
5. Reproducible reports and actionable, appropriately qualified findings.
