# Trustworthy Multi-Agent Healthcare AI Demonstrator

**Research demonstrator — not for clinical use.**

This application operationalizes a validated, locked, seven-agent
clinical-AI workflow (Input/Data Quality → Prediction → Explainability →
Trust & Fairness → Safety → CDS/Reporting → Orchestrator) developed and
audited in notebooks NB3 and NB5–NB9. It is a UI/application layer only —
it does not redesign, retrain, or reinterpret the underlying research.

## What this is not

- Not a clinical deployment system.
- Not a diagnostic tool.
- Not a source of individualized treatment recommendations.
- The model output is a **model-predicted probability**, not a calibrated
  individual clinical risk estimate.

## Governance guarantees

- The NB3 logistic regression model and its preprocessor are locked and
  loaded read-only, with sha256 integrity verification at every startup.
- The decision threshold (0.35) is read exclusively from NB3's metadata
  file — there is exactly one source of truth for it in this codebase.
- Missing patient values are never fabricated or silently imputed for
  clinical interpretation.
- The Safety Agent is a mandatory control point; downstream reporting
  cannot present a clinical interpretation when its gate is blocked.
- The explainability method is an exact logistic-regression coefficient
  contribution decomposition — it is not SHAP and is never described as
  SHAP.

## Status

**Phase 1 of 9** — project scaffold, governance/config module, locked
model loader with integrity verification, and a Streamlit shell showing
the fixed seven-agent pipeline as placeholders. No agent logic is
implemented yet.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
config/                 Static governance configuration (hashes, agent order, banner)
backend/
  governance.py          Single source of truth for threshold, hashes, agent order
  model_loader.py         Integrity-verified loading of the locked model/preprocessor
  agents/                 (added from Phase 2 onward)
artifacts/
  model/                  Locked NB3 model + preprocessor (.joblib)
  metadata/                NB3 model metadata (authoritative threshold source)
research/                Full notebooks + evidence (not imported at runtime) — added later
tests/                   (added in Phase 9)
app.py                   Streamlit entrypoint
```

