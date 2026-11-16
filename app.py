import os
import json
import math
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
CSV_PATH = os.path.join(BASE_DIR, "heart_disease_dataset.csv")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=PUBLIC_DIR, static_url_path="/public")

# ── Pre-Trained Mathematical Parameters (Instant Startup: 0.05ms) ───────────
# Trained on 80/20 Stratified Partition (Seed 42)
INTERCEPT = 0.13294003548619876
COEFS = [
    -0.12353498115791746, # age
     0.07939016009420108, # sex
    -0.20080872478906575, # blood_pressure
    -0.06459332060833445, # cholesterol
     0.028876751462732306,# bmi
     0.15056438337077357, # smoker
     0.04001861905568811, # diabetes
     0.04096844599009106, # physical_activity
    -0.0205751640906407,  # family_history
     0.0003722426666967595# stress_level
]
MEANS = [55.4675, 0.46, 132.8825, 220.97, 29.045, 0.535, 0.485, 2.0175, 0.4675, 4.9025]
SCALES = [14.318133389167743, 0.4983974317750858, 26.80958212561322, 43.966511119259835, 6.373093832668715, 0.49877349568717066, 0.4997749493522062, 1.4025668433269052, 0.4989426319728562, 2.5996141540620985]

feature_display_names = [
    "Age", "Sex (Male)", "Blood Pressure", "Cholesterol", "BMI",
    "Smoker", "Diabetes", "Physical Activity", "Family History", "Stress Level"
]

feature_importance_list = []
for name, coef in zip(feature_display_names, COEFS):
    feature_importance_list.append({
        "feature": name,
        "coef": round(float(coef), 4),
        "impact": "Higher Risk" if coef > 0 else "Lower Risk"
    })
feature_importance_list.sort(key=lambda x: abs(x["coef"]), reverse=True)

# Fast loading of 500 patient records using standard library csv module
dataset_records = []
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            hd = int(row["heart_disease"])
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
                "heart_disease": hd,
                "diagnosis_text": "Heart Disease" if hd == 1 else "Healthy / No Disease"
            })

METRICS = {
    "accuracy": "56.00%",
    "error_rate": "44.00%",
    "precision": "0.5616",
    "recall": "0.7736",
    "specificity": "0.3191",
    "f1_score": "0.6508",
    "mcc": "0.1043",
    "cohen_kappa": "0.0950",
    "log_loss": "0.7009",
    "roc_auc": "0.5054",
    "tp": 41,
    "fp": 32,
    "tn": 15,
    "fn": 12,
    "total_records": len(dataset_records) if dataset_records else 500,
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

        sample = [
            age, sex, blood_pressure, cholesterol, bmi,
            smoker, diabetes, physical_activity, family_history, stress_level
        ]

        # Fast StandardScaler z-score normalization
        scaled = [(sample[i] - MEANS[i]) / SCALES[i] for i in range(10)]
        
        # Logistic Regression log-odds score z
        z_score = INTERCEPT + sum(COEFS[i] * scaled[i] for i in range(10))
        
        # Sigmoid probability: P(Y=1) = 1 / (1 + e^-z)
        prob = (1.0 / (1.0 + math.exp(-z_score))) * 100.0
        pred_label = 1 if prob >= 50.0 else 0

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

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(PUBLIC_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
