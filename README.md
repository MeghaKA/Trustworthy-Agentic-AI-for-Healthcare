# Trustworthy Agentic AI for Healthcare

## Explainable and Safety-Aware Clinical Decision Support Using NHANES Data

A research project exploring how explainable machine learning, trustworthiness evaluation, safety governance, and multi-agent systems can support transparent healthcare decision-making workflows.

The project uses data from the *National Health and Nutrition Examination Survey (NHANES)* and focuses on diabetes-related early-risk prediction, model interpretability, input quality, fairness, and controlled clinical interpretation.

>**Research prototype**: This project is not a clinically validated diagnostic system. Its outputs are not medical diagnoses or treatment recommendations.

## Research Workflow

Data Integration
      ↓
Cohort and Feature Engineering
      ↓
Predictive Modelling
      ↓
Explainable AI
      ↓
Trust and Fairness Evaluation
      ↓
Explanation Consistency
      ↓
Clinical Decision-Support Prototype
      ↓
Multi-Agent Workflow
      ↓
End-to-End Validation

The research is organized into nine notebooks (NB1–NB9), progressively developed into a Streamlit-based demonstrator.

## Key Components

* *Predictive Modelling*: Locked Logistic Regression model for diabetes-related early-risk prediction.
* *Explainable AI*: Feature-contribution analysis and model interpretation.
* *Trust and Fairness*: Population-level performance and subgroup evaluation.
* *Explanation Consistency*: Analysis of explanation stability and consistency.
* *Safety Governance*: Input validation, interpretation gates, and evidence checks.
* *Multi-Agent Workflow*: Coordinated agents for prediction, explainability, trust evaluation, and safety.
* *Decision-Support Reporting*: Structured presentation of model outputs and supporting evidence.

## Repository Structure

data/             Raw data and data documentation
demonstrator/     Streamlit application and supporting artifacts
images/           Selected research figures
notebooks/        Nine-stage research workflow (NB1–NB9)
results/          Selected research evidence and validation outputs

## Data Source

The project uses public-use NHANES data from the *2017–March 2020 Pre-Pandemic cycle,* provided by the U.S. National Center for Health Statistics (NCHS), part of the Centers for Disease Control and Prevention (CDC).

Official source: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx

See data/raw/README.md for information about the source files and data/processed/README.md for the processing workflow.

## Research Demonstrator

The Streamlit demonstrator integrates the locked model and preprocessing pipeline with input-quality checks, deterministic feature contributions, trust and fairness evidence, safety gates, and structured reporting.

The application uses the existing model artifacts and does not retrain the model during inference.

## Explainability Methods

Notebook 4 explores SHAP-based feature-importance consistency. Notebook 6 and the deployed demonstrator use an exact Logistic Regression coefficient-by-transformed-feature contribution decomposition. These are distinct methods and are documented separately.

## Limitations and Responsible Use

This project is a research prototype, not a medical device or clinically validated decision-support system. Model outputs should not be interpreted as confirmed individual clinical risks. The model does not independently establish a diagnosis or recommend treatment.

## Author

*Megha K A*

MSc Data Analytics · B.Tech Computer Science and Engineering

Research interests: Trustworthy AI, Explainable AI, Healthcare AI, Clinical Decision Support, and AI Safety.

## License

This project is licensed under the MIT License. See LICENSE.
