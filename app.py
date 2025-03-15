import numpy as np
import joblib
from flask import Flask, request, render_template

# Load the trained model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    loan_status = None
    if request.method == 'POST':
        # Get input data from the form
        data = request.form

        # Extract values from the form
        credit_score = int(data['credit_score'])
        income = int(data['income'])
        employment_status = data['employment_status']
        loan_amount = int(data['loan_amount'])
        monthly_income = int(data['monthly_income'])
        monthly_employment_status = data['monthly_employment_status']
        number_of_dependents = int(data['number_of_dependents'])
        months_employed = int(data['months_employed'])

        # Encode categorical variables
        employment_status = 1 if employment_status.lower() == 'employed' else 0
        monthly_employment_status = 1 if monthly_employment_status.lower() == 'employed' else 0

        # Calculate debt-to-income ratio
        debt_to_income_ratio = loan_amount / income

        # Create input array (must match the features used for training)
        input_data = np.array([[credit_score, income, employment_status, loan_amount, monthly_income,
                                monthly_employment_status, number_of_dependents, months_employed, debt_to_income_ratio]])

        # Standardize the features using the saved scaler
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Set loan status based on prediction
        loan_status = "approved" if prediction == 1 else "denied"

    return render_template('index.html', loan_status=loan_status)

if __name__ == "__main__":
    app.run(debug=True)
