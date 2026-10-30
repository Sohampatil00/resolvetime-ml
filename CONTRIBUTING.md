# Contributing

ResolveTime is an educational regression project. Changes should preserve a clear distinction between demonstrations, validation results, and independent test measurements.

## Local checks

Use Python 3.11 and install `requirements-lock.txt` in a virtual environment, then run:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

For interface changes, launch `python -m streamlit run app.py --server.port 8502` after training a model. Verify that prediction works and changing an input clears the previous result. Avoid adding prediction fields that the fitted model does not use.

## Changes to the ML workflow

- Preserve one initial captured snapshot per incident. Do not fill its missing fields from later updates.
- Never use resolution timestamps, final outcomes, or future events as input predictors.
- Fit preprocessing and feature selection only on each training fold.
- Keep incidents and opening days together in chronological splits. Respect label-availability cutoffs.
- Select models using validation data. Compare final test performance against the median baseline, even when the learned model performs worse.
- If changing the feature schema, update its version, tests, documentation, and model compatibility checks. Retrain rather than reusing an incompatible artifact.
- Keep unresolved-case exclusions and uneven split populations visible in reports.
- Changes motivated by inspecting this test set need a new independent future holdout before being presented as unbiased improvements.

## Reporting issues

Include Python and package versions, the command or interface action, the full error, and a minimal reproducible example. Use the public UCI dataset or clearly fictional test inputs. Do not upload credentials or private company tickets.

## Data and generated files

Raw data, virtual environments, generated predictions, and additional model artifacts stay outside Git. The selected `models/resolution_model.joblib` is included for Streamlit cloud inference. If replacing it, keep its dependencies, reference evaluation, and presentation claims consistent. The dataset downloader records source attribution and hashes. Documentation may include aggregate evaluation snapshots, clearly identified as historical measurements rather than new test results.
