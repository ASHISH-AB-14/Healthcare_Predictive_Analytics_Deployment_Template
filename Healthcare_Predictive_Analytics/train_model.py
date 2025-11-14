import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load your dataset
df = pd.read_csv("diabetes.csv")    # <-- change dataset name here

# X = Features, y = Target
X = df.drop("Outcome", axis=1)      # <-- Change "Outcome" if your label name is different
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Create artifacts folder if not exists
os.makedirs("artifacts", exist_ok=True)

# Save model + scaler
joblib.dump(model, "artifacts/model_randomforest.joblib")
joblib.dump(scaler, "artifacts/scaler.joblib")

print("✔ Model and scaler saved successfully in artifacts/ folder!")
