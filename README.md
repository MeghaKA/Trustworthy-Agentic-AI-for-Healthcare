# Trustworthy Agentic AI for Healthcare

### Explainable and Safety-Aware Clinical Decision Support using NHANES Data

A research project exploring how **Explainable AI, trustworthy machine learning, safety governance, fairness evaluation, and multi-agent systems** can be combined to develop a transparent healthcare decision-support workflow.

The project uses **NHANES (National Health and Nutrition Examination Survey)** data and focuses on diabetes-related early-risk assessment, with emphasis on model interpretability, input quality, trustworthiness, and controlled clinical interpretation.

> **Research prototype:** This project is not a medical device and does not provide medical diagnosis, treatment recommendations, or autonomous clinical decisions.

---

## Research Workflow

```text
Data Integration
      ↓
Cohort & Target Definition
      ↓
Predictive Modelling
      ↓
Explainable AI
      ↓
Trust & Fairness Evaluation
      ↓
Explanation Consistency
      ↓
Clinical Decision Support
      ↓
Multi-Agent Healthcare Workflow
      ↓
Automated Research Audit

The workflow is developed through nine research notebooks (NB1–NB9) and progressively integrated into a Streamlit-based demonstrator.

⸻

###  Key Components

* Machine Learning — locked Logistic Regression model for diabetes-related early-risk prediction
* Explainable AI — deterministic feature-contribution analysis
* Trust & Fairness — population-level performance and subgroup evaluation
* Explanation Consistency — consistency analysis of model explanations
* Safety Governance — explicit input, interpretation, and evidence gates
* Multi-Agent Architecture — specialized agents for input quality, prediction, explainability, trust/fairness, and safety
* Clinical Decision Support — governed presentation of technical model outputs

⸻

###  Repository Structure

agents/          Research-level agent architecture
data/            Data documentation and dataset structure
demonstrator/    Streamlit research demonstrator
images/          Selected research and system figures
models/          Model artifacts and documentation
notebooks/       NB1–NB9 research workflow
results/         Selected research results and evidence

### Data

The project uses data from the National Health and Nutrition Examination Survey (NHANES) conducted by the U.S. National Center for Health Statistics (NCHS/CDC).

Raw NHANES files and participant-level processed datasets are not redistributed in this repository. The notebooks document the data-processing and modelling workflow.

⸻

## Demonstrator

A Streamlit-based research demonstrator operationalizes the audited workflow with:

* input quality validation
* locked model inference
* deterministic explanations
* trust and fairness evidence
* safety governance
* decision-support reporting

The demonstrator does not retrain or modify the locked research model.

⸻

## Research Focus

Trustworthy AI · Explainable AI · Healthcare AI · Clinical Decision Support · AI Safety · Fairness · Multi-Agent Systems

⸻

## Author

Megha K A
MSc Data Analytics | B.Tech Computer Science & Engineering

Research interests: Trustworthy AI, Explainable AI, Healthcare AI, Clinical Decision Support, and AI Safety.

⸻

## License

This project is licensed under the MIT License. See LICENSE⁠￼
