# Import necessary libraries
import joblib                  # For saving and loading models
import pandas as pd            # For data manipulation
import numpy as np             # For numerical operations
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical variables and scaling
from sklearn.model_selection import train_test_split             # For splitting dataset into train, validation, and test sets
from sklearn import metrics                                    # For evaluating model performance

# Load the dataset
df = pd.read_csv('../data/loan_approval_dataset.csv')
print(df.head())  # Display first few rows to inspect the dataset

# Clean column names (remove leading/trailing spaces) and drop irrelevant columns
df.columns = df.columns.str.strip()
df.drop(['luxury_assets_value', 'bank_asset_value', 'loan_id', 'education', 'self_employed'], axis=1, inplace=True)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.to_list()
numerical_cols = df.select_dtypes(include=['int64']).columns.to_list()

# Apply square root transformation to reduce skewness of financial features
df['residential_assets_value'] = np.sqrt(df['residential_assets_value'] - df['residential_assets_value'].min() + 1)
df['commercial_assets_value'] = np.sqrt(df['commercial_assets_value'] - df['commercial_assets_value'].min() + 1)
df['loan_amount'] = np.sqrt(df['loan_amount'] - df['loan_amount'].min() + 1)

# Encode the target variable: 'Approved' => 0, 'Rejected' => 1
le = LabelEncoder()
df['loan_status'] = le.fit_transform(df['loan_status'])

# Standardize all features except the target to have zero mean and unit variance
scaler = StandardScaler()
for col in df.columns.to_list():
    if col == 'loan_status':
        continue
    df[col] = scaler.fit_transform(df[[col]])

# Separate features (X) and target (y)
X = df.drop(columns='loan_status')
y = df['loan_status']

# Split the data into training (70%), validation (15%), and testing (15%) sets
# Use stratify=y to preserve class distribution
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_remaining, y_remaining, train_size=0.5, random_state=42, stratify=y_remaining
)

# Print the shapes of the train, validation, and test sets
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Load the pre-trained Decision Tree model
loaded_model = joblib.load("../model/decision_tree_model.pkl")

# Make predictions on the test set
y_test_pred = loaded_model.predict(X_test)

# Evaluate model performance on the test set
print("Classification report of Testing")
print(metrics.classification_report(y_test, y_test_pred, target_names=['Approved', 'Rejected']))
