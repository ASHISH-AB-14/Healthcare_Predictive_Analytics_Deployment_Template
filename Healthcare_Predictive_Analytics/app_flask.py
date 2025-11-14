# """
# Flask Web App for Healthcare Predictive Analytics (Disease Detection)
# ---------------------------------------------------------------------
# Run:
#     python app_flask.py
# Then open: http://127.0.0.1:5000
# """

from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load saved model and scaler
model_path = os.path.join("artifacts", "model_randomforest.joblib")
scaler_path = os.path.join("artifacts", "scaler.joblib")

model = joblib.load(model_path) if os.path.exists(model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form values
        data = [float(x) for x in request.form.values()]
        features = np.array([data])

        # Scale if scaler is available
        if scaler:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        result = {
            'prediction': int(prediction),
            'probability': round(float(proba), 3) if proba is not None else None
        }
        return render_template('index.html', prediction_text=f"Disease Risk: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
