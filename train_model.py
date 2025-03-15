import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
data = pd.read_csv('loan_data.csv')

# Create loan_approved column based on basic approval criteria
data['loan_approved'] = np.where(
    (data['credit_score'] >= 600) & (data['income'] >= 40000) & (data['employment_status'] == 'Employed'),
    1,  # Approved (1)
    0   # Denied (0)
)

# Feature Engineering
# Calculate debt-to-income ratio
data['debt_to_income_ratio'] = data['loan_amount'] / data['income']

# Preprocessing (encode categorical variables)
data['employment_status'] = data['employment_status'].map({'Employed': 1, 'Unemployed': 0})
data['monthly_employment_status'] = data['monthly_employment_status'].map({'Employed': 1, 'Unemployed': 0})

# Features (we'll use the debt-to-income ratio, among other features)
X = data[['credit_score', 'income', 'employment_status', 'loan_amount', 'monthly_income', 
          'monthly_employment_status', 'number_of_dependents', 'months_employed', 'debt_to_income_ratio']]

# Target (loan approval status)
y = data['loan_approved']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predictions on the test set
y_pred = model.predict(X_test_scaled)

# Model Evaluation using accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output model evaluation metrics
print("Model training complete!")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
