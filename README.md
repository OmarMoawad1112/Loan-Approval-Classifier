# Loan Approval Prediction

## 1️⃣ Project Overview
This project focuses on predicting whether a loan application will be approved using applicant demographics, financial attributes, and asset information from the **Kaggle Loan Approval dataset**.
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

## 3️⃣ Models & Techniques
- **Models used:** Logistic Regression, Decision Tree, Random Forest  
- **Techniques applied:**
  - Data preprocessing (handling missing values)
  - Encoding categorical variables
  - Feature scaling (StandardScaler)
  - Feature engineering (optional)
  - Hyperparameter tuning
  - Cross-validation

---

## 4️⃣ Results / Metrics
- **Performance metrics:** classification report (precision, recall, F1-score), confusion matrix  
- **Best performing model:** Decision Tree with accuracy (98% validation - 98% testing) 
---

## 5️⃣ Usage / Instructions
**Prerequisites:**
- Python 3.8+
- Required packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

**How to run:**
1. Clone the repository:  
   ```bash
     git clone https://github.com/OmarMoawad1112/Loan-Approval-Classifier.git
   ```
2. Navigate to the project directory:
   ```bash
     cd Loan-Approval-Classifier
   ```
3. Install dependencies:
   ```bash
     pip install -r requirements.txt
   ```
4. Open the Jupyter notebook:
   ```bash
     jupyter notebook notebook/loan_approval.ipynb
   ```
5. Run the notebook to see data analysis, model training, evaluation, and saving the Decision Tree model.
   
## 6️⃣ To load and use the model for predictions:
```bash
  import joblib
  loaded_model = joblib.load("models/decision_tree_model.pkl")
  preds = loaded_model.predict(X_test)
```
