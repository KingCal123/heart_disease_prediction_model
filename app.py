import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
CSV_PATH = os.path.join(BASE_DIR, "heart_disease_dataset.csv")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=PUBLIC_DIR, static_url_path="/public")

# ── Load Dataset & Train Estimator In-Memory ─────────────────────────────────
df = pd.read_csv(CSV_PATH)

mapping_binary = {"Yes": 1, "No": 0}
mapping_sex = {"Male": 1, "Female": 0}

df_encoded = df.copy()
df_encoded["sex_enc"] = df_encoded["sex"].map(mapping_sex)
df_encoded["smoker_enc"] = df_encoded["smoker"].map(mapping_binary)
df_encoded["diabetes_enc"] = df_encoded["diabetes"].map(mapping_binary)
df_encoded["family_history_enc"] = df_encoded["family_history"].map(mapping_binary)

feature_cols = [
    "age", "sex_enc", "blood_pressure", "cholesterol", "bmi",
    "smoker_enc", "diabetes_enc", "physical_activity", "family_history_enc", "stress_level"
]
feature_display_names = [
    "Age", "Sex (Male)", "Blood Pressure", "Cholesterol", "BMI",
    "Smoker", "Diabetes", "Physical Activity", "Family History", "Stress Level"
]

X = df_encoded[feature_cols].values
y = df_encoded["heart_disease"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train_scaled, y_train)

# Calculate exact test metrics
y_pred = classifier.predict(X_test_scaled)
y_prob = classifier.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Feature coefficients
coefs = classifier.coef_[0]
feature_importance_list = []
for name, coef in zip(feature_display_names, coefs):
    feature_importance_list.append({
        "feature": name,
        "coef": round(float(coef), 4),
        "impact": "Higher Risk" if coef > 0 else "Lower Risk"
    })
feature_importance_list.sort(key=lambda x: abs(x["coef"]), reverse=True)

# Prepare all 500 patient records for interactive table
dataset_records = []
for i, row in df.iterrows():
    dataset_records.append({
        "id": i + 1,
        "age": int(row["age"]),
        "sex": str(row["sex"]),
        "blood_pressure": int(row["blood_pressure"]),
        "cholesterol": int(row["cholesterol"]),
        "bmi": float(row["bmi"]),
        "smoker": str(row["smoker"]),
        "diabetes": str(row["diabetes"]),
        "physical_activity": int(row["physical_activity"]),
        "family_history": str(row["family_history"]),
        "stress_level": int(row["stress_level"]),
        "heart_disease": int(row["heart_disease"]),
        "diagnosis_text": "Heart Disease" if row["heart_disease"] == 1 else "Healthy / No Disease"
    })

METRICS = {
    "accuracy": f"{acc * 100:.2f}%",
    "error_rate": f"{(1.0 - acc) * 100:.2f}%",
    "precision": f"{prec:.4f}",
    "recall": f"{rec:.4f}",
    "specificity": f"{spec:.4f}",
    "f1_score": f"{f1:.4f}",
    "mcc": f"{mcc:.4f}",
    "cohen_kappa": f"{kappa:.4f}",
    "log_loss": f"{loss:.4f}",
    "roc_auc": f"{auc:.4f}",
    "tp": int(tp),
    "fp": int(fp),
    "tn": int(tn),
    "fn": int(fn),
    "total_records": len(df),
    "feature_importances": feature_importance_list,
    "dataset_json": json.dumps(dataset_records)
}

@app.route("/")
def index():
    return render_template("index.html", **METRICS)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        
        age = float(data.get("age", 55))
        sex_raw = str(data.get("sex", "Male")).capitalize()
        sex = 1.0 if sex_raw == "Male" else 0.0
        
        blood_pressure = float(data.get("blood_pressure", 130))
        cholesterol = float(data.get("cholesterol", 210))
        bmi = float(data.get("bmi", 27.5))
        
        smoker_raw = str(data.get("smoker", "No")).capitalize()
        smoker = 1.0 if smoker_raw == "Yes" else 0.0
        
        diabetes_raw = str(data.get("diabetes", "No")).capitalize()
        diabetes = 1.0 if diabetes_raw == "Yes" else 0.0
        
        physical_activity = float(data.get("physical_activity", 2))
        
        family_history_raw = str(data.get("family_history", "No")).capitalize()
        family_history = 1.0 if family_history_raw == "Yes" else 0.0
        
        stress_level = float(data.get("stress_level", 5))

        sample = np.array([[
            age, sex, blood_pressure, cholesterol, bmi,
            smoker, diabetes, physical_activity, family_history, stress_level
        ]])

        sample_scaled = scaler.transform(sample)
        z_score = float(classifier.decision_function(sample_scaled)[0])
        prob = float(classifier.predict_proba(sample_scaled)[0][1]) * 100.0
        pred_label = int(classifier.predict(sample_scaled)[0])

        if prob >= 65.0:
            risk_tier = "High Risk"
            risk_color = "#ef4444"
            summary_message = "Patient exhibits pronounced clinical risk factors strongly associated with cardiovascular disease."
        elif prob >= 45.0:
            risk_tier = "Moderate Risk"
            risk_color = "#f59e0b"
            summary_message = "Patient demonstrates borderline indicators. Lifestyle modification and clinical monitoring advised."
        else:
            risk_tier = "Low Risk"
            risk_color = "#10b981"
            summary_message = "Cardiovascular profile appears favorable within normative parameters."

        # Compute individualized risk driver factors
        individual_factors = []
        if blood_pressure >= 140:
            individual_factors.append("Elevated Blood Pressure (Hypertension Stage 2)")
        elif blood_pressure >= 130:
            individual_factors.append("Stage 1 Hypertension")
            
        if cholesterol >= 240:
            individual_factors.append("High Serum Cholesterol (> 240 mg/dL)")
        elif cholesterol >= 200:
            individual_factors.append("Borderline High Cholesterol (200-239 mg/dL)")

        if bmi >= 30.0:
            individual_factors.append(f"Classified Obesity (BMI: {bmi:.1f})")
        elif bmi >= 25.0:
            individual_factors.append(f"Overweight Category (BMI: {bmi:.1f})")

        if smoker == 1.0:
            individual_factors.append("Active Tobacco Smoker")
        if diabetes == 1.0:
            individual_factors.append("Diagnosed Diabetic Condition")
        if family_history == 1.0:
            individual_factors.append("Positive Family History of Cardiovascular Disease")
        if stress_level >= 7:
            individual_factors.append(f"Severe Stress Metric (Score: {int(stress_level)}/9)")
        if physical_activity == 0:
            individual_factors.append("Sedentary Lifestyle (0 hrs physical activity/wk)")

        if not individual_factors:
            individual_factors.append("No critical clinical risk elevations detected.")

        return jsonify({
            "status": "success",
            "prediction": "Heart Disease Risk" if pred_label == 1 else "Low Risk / Negative",
            "prediction_code": pred_label,
            "probability": round(prob, 1),
            "log_odds_score": round(z_score, 3),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "summary_message": summary_message,
            "contributing_factors": individual_factors,
            "inputs": {
                "age": age,
                "sex": sex_raw,
                "blood_pressure": blood_pressure,
                "cholesterol": cholesterol,
                "bmi": bmi,
                "smoker": smoker_raw,
                "diabetes": diabetes_raw,
                "physical_activity": physical_activity,
                "family_history": family_history_raw,
                "stress_level": stress_level
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Route for static files / images in development
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(PUBLIC_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
