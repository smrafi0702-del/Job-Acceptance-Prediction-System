# 🎯 Job Acceptance Prediction System

An end-to-end HR analytics & machine learning project that predicts whether a candidate will accept or reject a job offer, using a realistic, intentionally messy recruitment dataset.
Developed as part of the HCL GUVI Data Science & Machine Learning professional program (Batch DS-S-WE-T-B158) by **MUHAMMAD RAFI S.M.** · 2026

---

## 📌 Problem Statement

Recruitment and placement teams deal with large volumes of candidate data — academic performance, skills, experience, interview outcomes, and job market conditions. Not every eligible candidate who receives an offer ends up accepting it.

**Goal:** Analyze candidate placement data and build a Job Acceptance Prediction System that can:

- Predict whether a candidate will accept or reject a job offer.
- Identify the key factors influencing that decision.
- Handle real-world data problems (missing values, duplicates, noisy/inconsistent categorical data).
- Provide actionable insights for recruitment teams.

---

## 💼 Business Use Cases

- Predict offer-acceptance probability before extending an offer.
- Reduce offer dropouts and shorten hiring cycle time.
- Identify which factors most affect placement success.
- Optimize interview and evaluation strategies.
- Support data-driven decisions for HR managers.

---

## 🗂️ Project Structure

```text
Job_Acceptance_Prediction_System/
│
├── 📁 raw/                               # Raw dataset folder
├── 📄 cleaned_data.csv                   # Output of cleaning phase
├── 📄 job_acceptance_missing_values_handled.csv # Missing values imputed dataset
├── 📄 job_acceptance_after_eda.csv       # Dataset post-EDA
├── 📄 feature_engineered_data.csv        # Engineered feature matrix
├── 📄 cleaned_job_placement_data_after_ml.csv # Final dataset used by model & app
├── 📄 model.pkl                          # Trained Random Forest model + metadata package
│
├── 📄 streamlit_app.py                   # Interactive 4-tab Streamlit HR dashboard
├── 📄 4_feature_engineering.ipynb        # Derived features notebook
└── 📄 5_ml_modeling.ipynb                # ML training & evaluation notebook
```

## 📊 Dataset Overview

Raw Dataset: Contains 50,000+ candidate records mimicking real-world recruitment patterns.

Intentional Data Quality Issues Handled: Missing values across multiple columns, duplicate-like records, inconsistent text spacing/casing, and rare logical anomalies.

Feature Categories Covered:

Academic Performance: SSC (10th), HSC (12th), Degree percentages, and Specializations.

Skills & Certifications: Technical scores, skills match percentage, and certifications count.

Work Experience: Years of experience, internships, relevant experience, previous CTC, and employment gaps.

Interview Performance: Technical, aptitude, and communication evaluation scores.

Job Market Conditions: Company tier, job role match, competition level, bond requirements, notice period, and relocation willingness.

Target Outcome (status): Placed (Job Accepted = 1) vs. Not Placed (Job Rejected = 0).

## 🤖 Machine Learning Model

Algorithm: Random Forest Classifier — chosen for robust handling of tabular data and strong feature explainability.

Target Variable: status (Placed = 1, Not Placed = 0).

Artifact: Serialized into model.pkl, containing the fitted classifier, scaling parameters, and exact training feature column order for seamless real-time inference.

## 🖥️ Streamlit Dashboard

An interactive, professional HR analytics web application (streamlit_app.py) featuring four distinct tabs:

🏢 Overview: Core project summary, key metrics, business use cases, and workflow introduction.

🔮 Predict Acceptance: Real-time form allowing recruiters to input candidate attributes and instantly evaluate offer acceptance probability with a confidence gauge.

📈 Dashboard & KPIs: 8 styled KPI cards (Total Candidates, Placement Rate, Acceptance Rate, Dropout Rate, High-Risk %, etc.) alongside interactive analytics charts.

⚡ Model Insights: Feature importance rankings highlighting the key drivers behind model predictions.

## 🚀 How to Run the App Locally

From inside your project directory, follow these steps:

Install required libraries:

pip install streamlit pandas numpy scikit-learn plotly
Run the Streamlit application:

streamlit run streamlit_app.py
Open the app: Open your browser and navigate to http://localhost:8501. (Note: Because model.pkl and the cleaned dataset sit in the same folder as streamlit_app.py, no path changes are required).

## 📦 Deliverables

Source Code: Python scripts (streamlit_app.py) and Jupyter notebooks for preprocessing, feature engineering, and ML modeling.

SQL Database: Cleaned dataset stored and connected via SQLAlchemy.

Streamlit Application: Interactive multi-tab HR Analytics dashboard.

Trained Model: Serialized model.pkl package.

Documentation: Comprehensive project README.
