# Raw NHANES Data

This directory contains the public-use NHANES source files used by Notebook 1 for data integration and preprocessing.

## Data Source

National Health and Nutrition Examination Survey (NHANES), 2017–March 2020 Pre-Pandemic cycle

Provided by the National Center for Health Statistics (NCHS), part of the U.S. Centers for Disease Control and Prevention (CDC).

Official source: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx

## Included Files

The following source components are used by Notebook 1:

* P_ALB_CR.xpt — Urine Albumin and Creatinine
* P_ALQ.xpt — Alcohol Use
* P_BIOPRO.xpt — Standard Biochemistry Profile
* P_BMX.xpt — Body Measures
* P_BPXO.xpt — Blood Pressure
* P_CBC.xpt — Complete Blood Count
* P_DBQ.xpt — Diet Behavior and Nutrition
* P_DEMO.xpt — Demographics
* P_DIQ.xpt — Diabetes Questionnaire
* P_GHB.xpt — Glycohemoglobin
* P_GLU.xpt — Plasma Fasting Glucose
* P_HDL.xpt — HDL Cholesterol
* P_HIQ.xpt — Health Insurance
* P_HSCRP.xpt — High-Sensitivity C-Reactive Protein
* P_HUQ.xpt — Hospital Utilization and Access to Care
* P_PAQ.xpt — Physical Activity
* P_RXQ_RX.xpt — Prescription Medications
* P_SLQ.xpt — Sleep Disorders
* P_SMQ.xpt — Smoking — Cigarette Use
* P_TCHOL.xpt — Total Cholesterol
* P_TRIGLY.xpt — Triglycerides

## Usage

Notebook 1 integrates these source files using the NHANES participant identifier (SEQN) and generates the master dataset used by subsequent notebooks.

The processed datasets generated during the research workflow are not stored in this directory.

## Data Note

These are public-use survey data, not private clinical records. Please consult the official NHANES documentation for survey design, variable definitions, and appropriate interpretation.
