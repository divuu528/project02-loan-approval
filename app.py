import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load Model + Scaler
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

dec_target = {1: "Approved", 0: "Rejected"}

@app.route("/")
def home():
    return render_template("loan_data_dashboard.html", result=None, input_data={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs
        form_data_str = {
            'no_of_dependents': request.form.get('no_of_dependents'),
            'education': request.form.get('education'),
            'self_employed': request.form.get('self_employed'),
            'income_annum': request.form.get('income_annum'),
            'loan_amount': request.form.get('loan_amount'),
            'loan_term': request.form.get('loan_term'),
            'cibil_score': request.form.get('cibil_score'),
            'residential_assets_value': request.form.get('residential_assets_value'),
            'commercial_assets_value': request.form.get('commercial_assets_value'),
            'luxury_assets_value': request.form.get('luxury_assets_value'),
            'bank_asset_value': request.form.get('bank_asset_value')
        }

        # Convert to numeric
        form_data_num = {k: int(v) for k, v in form_data_str.items()}

        df = pd.DataFrame([form_data_num])
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)[0]
        result = dec_target[prediction]

        return render_template("loan_data_dashboard.html",
                               result=result,
                               input_data=form_data_str)
    except Exception as e:
        return render_template("loan_data_dashboard.html",
                               result=f"Error: {e}",
                               input_data=form_data_str)
                               
if __name__ == "__main__":
    app.run(debug=True)

