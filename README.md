# Loan Approval Prediction

## 1️⃣ Project Overview
This project focuses on predicting whether a loan application will be approved using applicant demographics, financial attributes, and asset information from the **Kaggle Loan Approval dataset**. The dataset contains features such as the **number of dependents**, **education**, **employment status**, **annual income**, **loan amount**, **loan term**, **CIBIL score**, and various asset values, which together help determine the likelihood of loan approval.

**Objectives:**
- Understand the relationship between applicant demographics, financial profiles, and asset holdings on loan approval decisions.
- Build and compare **Logistic Regression**, **Decision Tree**, and **Random Forest models** to predict loan approval (`loan_status`).
- Evaluate model performance using classification metrics, confusion matrices, precision, recall, and F1-score.
- Apply data preprocessing, encoding, feature scaling, and hyperparameter tuning to improve model accuracy and robustness.

---

## 2️⃣ Dataset Information
- **Source:** [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset?resource=download)
- **Shape:** 4269 rows, 13 columns
- **Features:**
  - `no_of_dependents`
  - `education`
  - `self_employed`
  - `income_annum`
  - `loan_amount`
  - `loan_term`
  - `cibil_score`
  - `residential_assets_value`
  - `commercial_assets_value`
  - `luxury_assets_value`
  - `bank_asset_value`
- **Target:** `loan_status`

---

## 3️⃣ Objectives / Goals
- Understand factors influencing loan approval decisions.
- Build and compare **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers.
- Evaluate model performance using classification metrics: accuracy, precision, recall, F1-score, and confusion matrices.
- Apply preprocessing, feature scaling, encoding, and hyperparameter tuning to improve model performance and robustness.

---

## 4️⃣ Models & Techniques
- **Models used:** Logistic Regression, Decision Tree, Random Forest  
- **Techniques applied:**
  - Data preprocessing (handling missing values)
  - Encoding categorical variables
  - Feature scaling (StandardScaler)
  - Feature engineering (optional)
  - Hyperparameter tuning
  - Cross-validation

---

## 5️⃣ Results / Metrics
- **Performance metrics:** classification report (precision, recall, F1-score), confusion matrix  
- **Best performing model:** Decision Tree  

---

## 6️⃣ Folder Structure
