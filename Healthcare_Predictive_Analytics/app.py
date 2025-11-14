from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler
model = joblib.load("artifacts/model_randomforest.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect patient data
        data = [
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["bmi"]),
            float(request.form["bp"]),
            float(request.form["glucose"]),
            float(request.form["oxygen"]),
            float(request.form["cholesterol"]),
            float(request.form["heartrate"])
        ]

        features = np.array([data])

        # Apply scaler
        features = scaler.transform(features)

        # Prediction
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if prediction == 1:
            message = "⚠ High Risk – Please consult a doctor."
            color = "danger"
        else:
            message = "✔ Low Risk – You are safe."
            color = "success"

        return render_template(
            "index.html",
            result_message=message,
            result_color=color,
            probability=round(proba * 100, 2)
        )

    except Exception as e:
        return render_template("index.html", result_message=str(e), result_color="warning")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
