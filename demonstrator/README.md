# Trustworthy Multi-Agent Healthcare AI Demonstrator

**Research demonstrator — not for clinical use.**
This application progressively operationalizes a locked, research-audited multi-agent clinical-AI workflow (Input/Data Quality → Prediction → Explainability → Trust & Fairness → Safety → CDS/Reporting → Orchestrator) developed and audited in notebooks NB3 and NB5–NB9. It is a UI/application layer only — it does not redesign, retrain, or reinterpret the underlying research.

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

## Execution order and the two-tier safety gate

The implemented pipeline runs in this fixed order:

```
Input Quality Agent
    -> Prediction Agent
    -> Explainability Agent
    -> Trust/Fairness Agent
    -> Safety Agent
    -> Display
```

There are two distinct gates in this pipeline, and they serve different
purposes:

- **The Prediction Agent (Phase 3) is a compute-time hard block.** If the
  Input/Data Quality Agent reports `safety_gate == INVALID_INPUT_SCHEMA`,
  the Prediction Agent refuses to call `preprocessor.transform()` or
  `model.predict_proba()` at all — no computation is attempted on a
  structurally invalid input. This is verified with method-spy tests.
- **The Safety Agent (Phase 6) is the final, authoritative downstream
  display gate.** It runs after all four upstream agents have already
  executed (or refused to execute), and every `*_allowed` flag it
  produces is the logical AND of its own rule and the corresponding
  upstream flag(s) — so it can only agree with or further restrict what
  upstream already decided, never loosen it. The Streamlit UI
  (`app.py`) consults the Safety Agent's flags directly before
  rendering the Prediction and Explainability sections, not only each
  agent's own internal flags, which is what makes the Safety Agent
  impossible to bypass through the normal UI workflow.

## Status

**Phases 1–6 of 9 implemented** — artifact loading/integrity, patient
input form, Input/Data Quality Agent, Prediction Agent, Explainability
Agent, Trust/Fairness Agent, and the Safety Agent (mandatory governance
control point). CDS/Reporting and the Orchestrator (Phases 7–8) are not
yet implemented.

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
  schema.py                Authoritative 214-feature schema + curated UI-exposed subset
  agents/
    input_quality_agent.py   Phase 2 — NB9 live schema/completeness validator
    prediction_agent.py      Phase 3 — locked model inference (transform-only)
    explainability_agent.py  Phase 4 — additive logistic coefficient decomposition
    trust_fairness_agent.py  Phase 5 — static NB5/NB6 population-level evidence
    safety_agent.py           Phase 6 — final governance/display gate
ui/
  patient_input_form.py   Structured Streamlit form (30-field curated subset)
artifacts/
  model/                  Locked NB3 model + preprocessor (.joblib)
  metadata/                NB3 model metadata (authoritative threshold source)
  schema/                  Locked 214-feature schema CSV (verbatim from NB7 export)
  evidence/                Static NB5/NB6 population-level evidence CSVs (verbatim)
research/                Full notebooks + evidence (not imported at runtime) — added later
tests/                   (added in Phase 9)
app.py                   Streamlit entrypoint
```
