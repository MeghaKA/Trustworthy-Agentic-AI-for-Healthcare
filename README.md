# Trustworthy Agentic AI for Healthcare

### Explainable and Safety-Aware Clinical Decision Support Using NHANES Data

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/AI-Explainable%20%26%20Trustworthy-5B6DAA" alt="Explainable and Trustworthy AI"/>
  <img src="https://img.shields.io/badge/Research-Healthcare%20AI-287D8E" alt="Healthcare AI Research"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"/>
</p>

A research project investigating how *explainable machine learning, trustworthiness evaluation, safety governance, fairness analysis, and multi-agent systems* can support transparent healthcare decision-support workflows.

Using public-use data from the *National Health and Nutrition Examination Survey (NHANES)* the project focuses on diabetes-related early-risk prediction and the responsible interpretation of model outputs.

*Research prototype:* This project is not a clinically validated diagnostic system, medical device, or substitute for professional medical judgement. Its outputs must not be used independently for diagnosis or treatment decisions.

⸻

## Research Workflow

The research is organized into nine notebooks (NB1–NB9), progressing from data preparation to end-to-end system validation.

Stage	Research component
NB1	Data integration and preprocessing
NB2	Cohort definition and feature engineering
NB3	Predictive modelling and clinical performance evaluation
NB4	Explainable AI and patient-level interpretation
NB5	Trust, fairness, and safety evaluation
NB6	Explainability consistency framework
NB7	Clinical decision-support prototype
NB8	Trustworthy multi-agent workflow
NB9	Automated integration and integrity audit

The validated research workflow is integrated into a Streamlit-based demonstrator.

## Key Components

* Predictive modelling: A locked logistic regression model with a fixed preprocessing pipeline and documented decision threshold.
* Explainable AI: Feature-importance exploration and deterministic model-contribution analysis.
* Trust and fairness: Aggregate model-performance and subgroup evaluations.
* Explanation consistency: Assessment of explanation similarity and stability.
* Safety governance: Input-quality checks, controlled interpretation, and evidence-based reporting gates.
* Multi-agent architecture: Specialized agents for input quality, prediction, explainability, trust and fairness, safety, and reporting.
* System validation: Automated integration checks and model-integrity verification.

## Explainability Methods

Notebook 4 explores SHAP-based feature-importance consistency. Notebook 6 and the deployed demonstrator use an exact logistic-regression coefficient-by-transformed-feature contribution decomposition.

These are distinct methods. The deployed demonstrator’s contribution analysis is not SHAP, and explanation consistency alone does not establish clinical validity.

Demonstrator

The Streamlit application brings the research components together in a governed workflow, including:

* Input-quality validation
* Locked model inference
* Deterministic explanation reconstruction
* Reference-based trust and fairness evidence
* Safety-gated interpretation
* Decision-support reporting

The demonstrator does not retrain the locked model. Model outputs are technical predictions, not established individual clinical-risk estimates.

Repository Structure

data/
├── raw/              # NHANES source files and provenance
└── processed/        # Processing documentation
demonstrator/
├── artifacts/        # Locked model, preprocessor, metadata and schema
├── backend/          # Model and governance agents
├── config/           # Governance configuration
└── ui/               # Patient-input interface
images/               # Selected research figures
notebooks/            # Nine-stage research workflow
results/              # Selected research evidence

## Data Source

The project uses public-use NHANES 2017–March 2020 Pre-Pandemic data provided by the U.S. National Center for Health Statistics (NCHS), part of the Centers for Disease Control and Prevention (CDC).

* Official NHANES data and documentation
* data/raw/README.md — Source files and provenance
* data/processed/README.md — Processing documentation

The notebooks document the data-processing workflow. Generated participant-level datasets and intermediate outputs are not committed to the repository.

Research Interests

Trustworthy AI · Explainable AI · Healthcare AI · Clinical Decision Support · AI Safety · Fairness · Multi-Agent Systems

Author

Megha K A

MSc Data Analytics · B.Tech Computer Science and Engineering

Research interests: Trustworthy AI, Explainable AI, Healthcare AI, Clinical Decision Support, and AI Safety.

License

This project is licensed under the MIT License.

