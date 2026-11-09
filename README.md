# Cardiovascular Disease Risk Classification & Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://palletsprojects.com/p/flask/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black.svg)](https://vercel.com/)

**Course**: AI and Expert Systems (Assignment 2)  
**Student**: Caleb Tetteh Martey (`PUIT/23110056`)  
**Programme**: BSc Information Technology  

---

## 📌 Project Overview
This project delivers an end-to-end Machine Learning pipeline and an interactive web-based clinical decision support system for predicting the likelihood of cardiovascular (heart) disease. Using a structured dataset of 500 clinical patient instances across 10 physiological risk indicators, the system trains a standardized **Logistic Regression** probability model and benchmarks it against a **Random Forest Classifier**.

### Key Highlights
* **Dataset**: 500 patient records, 10 clinical features (6 numerical, 4 categorical), 0 missing values.
* **Balanced Cohort**: 266 Heart Disease Cases (53.2%) vs. 234 Healthy Patients (46.8%).
* **Medical Screening Priority**: The Logistic Regression model achieves **77.36% Medical Recall (Sensitivity)** on the stratified hold-out test set, capturing 41 out of 53 active cardiovascular cases to minimize critical false-negative diagnostic outcomes.
* **Continuous Probabilistic Transfer**: Utilizes the non-linear Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$ to translate multi-dimensional log-odds risk scores into calibrated probability percentages.
* **Interactive Web Application**: Full-stack Flask application featuring real-time risk inference, animated probability gauge meters, visual analytics gallery (scatter plots, confusion matrix, ROC curves, sigmoid curve), and a 500-record dataset explorer table.

---

## 🔬 Clinical Predictor Features
1. `age`: Patient chronological age (years, range: 30–79)
2. `sex`: Biological sex (`Male`, `Female`)
3. `blood_pressure`: Resting blood pressure (mmHg, range: 90–179)
4. `cholesterol`: Serum cholesterol (mg/dL, range: 150–299)
5. `bmi`: Body Mass Index (kg/m², range: 18.1–40.0)
6. `smoker`: Active tobacco smoker status (`Yes`, `No`)
7. `diabetes`: Diagnosed diabetic condition (`Yes`, `No`)
8. `physical_activity`: Weekly physical activity level (hours/week: 0–4)
9. `family_history`: Family history of cardiovascular disease (`Yes`, `No`)
10. `stress_level`: Perceived psychological stress scale (1–9)
* **Target** (`heart_disease`): `1` = Heart Disease Present, `0` = Healthy / Unaffected

---

## 📊 Model Performance Summary (80/20 Stratified Test Set)

| Metric | Logistic Regression | Random Forest Classifier | Clinical Significance |
| :--- | :---: | :---: | :--- |
| **Recall / Sensitivity** | **77.36%** | 54.72% | Primary medical triage metric (caught 41/53 disease cases) |
| **Precision (PPV)** | **56.16%** | 51.79% | True positive rate among flagged patients |
| **Overall Accuracy** | **56.00%** | 49.00% | Global classification accuracy |
| **Specificity (TNR)** | 31.91% | 42.55% | Identification of disease-free cohort |
| **F1-Score** | **0.6508** | 0.5321 | Harmonic balance between precision and recall |
| **Log Loss** | **0.7009** | 0.7240 | Cross-entropy penalization on probability distributions |
| **ROC-AUC Score** | 0.5054 | 0.5217 | Discriminatory threshold area |

---

## 📈 Visual Analytics & Plots Generated
* **Figure 1**: `scatter_age_cholesterol.png` – Age vs. Serum Cholesterol bi-variate scatter plot.
* **Figure 2**: `scatter_bp_bmi.png` – Resting Blood Pressure vs. Body Mass Index scatter plot.
* **Figure 3**: `logistic_regression_sigmoid.png` – Sigmoid S-curve showing test patients and decision boundary ($p = 0.5$).
* **Figure 4**: `confusion_matrix.png` – Heatmap detailing TP (41), FN (12), FP (32), and TN (15).
* **Figure 5**: `roc_curves.png` – Receiver Operating Characteristic performance comparison.
* **Figure 6**: `feature_importance.png` – Signed log-odds coefficients illustrating clinical risk factors.

---

## 🚀 Local Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KingCal123/heart_disease_prediction_model.git
   cd heart_disease_prediction_model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run model training & figure generation:**
   ```bash
   python heart_disease_classification.py
   ```

4. **Start the web application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## ☁️ Vercel Deployment
The repository is structured for serverless execution on Vercel:
* `api/index.py`: Python WSGI entry point.
* `vercel.json`: Serverless routing configuration.
* `public/`: Static image routing for high-resolution plots.

---

## 📄 Academic Report
The complete technical coursework report can be generated locally using:
```bash
python generate_report.py
```
This generates `Assignment_Report_Caleb_Tetteh_Martey.docx` with all embedded figures, mathematical formulations, and clinical analysis.
