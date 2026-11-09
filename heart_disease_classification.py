"""
Heart Disease Classification & Comprehensive Evaluation Pipeline
AI and Expert Systems - Assignment 2

This script:
  1. Loads and preprocesses the heart disease dataset (500 records, 10 features).
  2. Encodes binary categorical variables (sex, smoker, diabetes, family_history).
  3. Performs an 80/20 stratified train/test split.
  4. Trains both a Logistic Regression model and a Random Forest Classifier.
  5. Computes full evaluation metrics:
     - Accuracy & Error Rate
     - Precision, Recall (Sensitivity), Specificity
     - F1-Score, MCC, Cohen's Kappa
     - Log Loss & ROC-AUC
  6. Generates high-resolution visualization figures:
     - scatter_age_cholesterol.png (Age vs Cholesterol colored by diagnosis)
     - scatter_bp_bmi.png (Blood Pressure vs BMI colored by diagnosis)
     - logistic_regression_sigmoid.png (Sigmoid probability curve)
     - confusion_matrix.png (Confusion matrix heatmap)
     - roc_curves.png (ROC-AUC performance curve)
     - feature_importance.png (Coefficients / Feature Importances)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "heart_disease_dataset.csv")

def run_pipeline():
    print("=" * 75)
    print("      HEART DISEASE CLASSIFICATION & EVALUATION PIPELINE (ASSIGNMENT 2)      ")
    print("=" * 75)

    # 1. Load Data
    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Encode Categorical Variables
    df_encoded = df.copy()
    mapping_binary = {"Yes": 1, "No": 0}
    mapping_sex = {"Male": 1, "Female": 0}

    df_encoded["sex"] = df_encoded["sex"].map(mapping_sex)
    df_encoded["smoker"] = df_encoded["smoker"].map(mapping_binary)
    df_encoded["diabetes"] = df_encoded["diabetes"].map(mapping_binary)
    df_encoded["family_history"] = df_encoded["family_history"].map(mapping_binary)

    feature_cols = [
        "age", "sex", "blood_pressure", "cholesterol", "bmi",
        "smoker", "diabetes", "physical_activity", "family_history", "stress_level"
    ]
    target_col = "heart_disease"

    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Split data: {len(X_train)} training samples, {len(X_test)} testing samples")

    # 4. Train Models
    # Model A: Logistic Regression (with StandardScaler for numerical stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    # Model B: Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Predictions
    # Logistic Regression
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

    # 5. Compute Comprehensive Metrics for Logistic Regression
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    tn, fp, fn, tp = cm_lr.ravel()

    acc_lr = accuracy_score(y_test, y_pred_lr)
    err_lr = 1.0 - acc_lr
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr) # Sensitivity
    spec_lr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
    f1_lr = f1_score(y_test, y_pred_lr)
    mcc_lr = matthews_corrcoef(y_test, y_pred_lr)
    kappa_lr = cohen_kappa_score(y_test, y_pred_lr)
    loss_lr = log_loss(y_test, y_prob_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    # Metrics for Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    spec_rf = tn_rf / (tn_rf + fp_rf) if (tn_rf + fp_rf) > 0 else 0.0
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    print("\n" + "=" * 50)
    print("      LOGISTIC REGRESSION PERFORMANCE METRICS      ")
    print("=" * 50)
    print(f"Accuracy:                    {acc_lr * 100:.2f}%")
    print(f"Error Rate:                  {err_lr * 100:.2f}%")
    print(f"Precision (PPV):             {prec_lr:.4f} ({prec_lr * 100:.2f}%)")
    print(f"Recall / Sensitivity (TPR):  {rec_lr:.4f} ({rec_lr * 100:.2f}%)")
    print(f"Specificity (TNR):           {spec_lr:.4f} ({spec_lr * 100:.2f}%)")
    print(f"F1-Score:                    {f1_lr:.4f}")
    print(f"Matthews Corr Coeff (MCC):   {mcc_lr:.4f}")
    print(f"Cohen's Kappa:               {kappa_lr:.4f}")
    print(f"Log Loss:                    {loss_lr:.4f}")
    print(f"ROC-AUC Score:               {auc_lr:.4f}")
    print("\nConfusion Matrix (LR):")
    print(f"  True Positives (TP):  {tp} | False Negatives (FN): {fn}")
    print(f"  False Positives (FP): {fp} | True Negatives (TN):  {tn}")

    print("\n" + "=" * 50)
    print("      RANDOM FOREST PERFORMANCE COMPARISON      ")
    print("=" * 50)
    print(f"Accuracy:                    {acc_rf * 100:.2f}%")
    print(f"Precision:                   {prec_rf:.4f}")
    print(f"Recall (Sensitivity):        {rec_rf:.4f}")
    print(f"Specificity:                 {spec_rf:.4f}")
    print(f"F1-Score:                    {f1_rf:.4f}")
    print(f"ROC-AUC:                     {auc_rf:.4f}")

    # 6. Generate Figures
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: Scatter Plot 1 - Age vs. Cholesterol
    plt.figure(figsize=(9, 6), dpi=300)
    palette = {0: "#2563eb", 1: "#dc2626"}
    markers = {0: "o", 1: "s"}
    sns.scatterplot(
        data=df,
        x="age",
        y="cholesterol",
        hue="heart_disease",
        style="heart_disease",
        palette=palette,
        markers=markers,
        s=80,
        alpha=0.85,
        edgecolor="w"
    )
    plt.title("Scatter Plot: Patient Age vs. Serum Cholesterol Level", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Age (years)", fontsize=12)
    plt.ylabel("Serum Cholesterol (mg/dL)", fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["No Disease (0)", "Heart Disease (1)"], title="Diagnosis", loc="upper left")
    plt.tight_layout()
    scatter1_path = os.path.join(BASE_DIR, "scatter_age_cholesterol.png")
    plt.savefig(scatter1_path)
    plt.close()
    print(f"[OK] Saved scatter plot 1: {scatter1_path}")

    # Figure 2: Scatter Plot 2 - Blood Pressure vs. BMI
    plt.figure(figsize=(9, 6), dpi=300)
    sns.scatterplot(
        data=df,
        x="blood_pressure",
        y="bmi",
        hue="heart_disease",
        style="heart_disease",
        palette=palette,
        markers=markers,
        s=80,
        alpha=0.85,
        edgecolor="w"
    )
    plt.title("Scatter Plot: Resting Blood Pressure vs. Body Mass Index (BMI)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Resting Blood Pressure (mmHg)", fontsize=12)
    plt.ylabel("Body Mass Index (BMI, kg/m²)", fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=["No Disease (0)", "Heart Disease (1)"], title="Diagnosis", loc="upper left")
    plt.tight_layout()
    scatter2_path = os.path.join(BASE_DIR, "scatter_bp_bmi.png")
    plt.savefig(scatter2_path)
    plt.close()
    print(f"[OK] Saved scatter plot 2: {scatter2_path}")

    # Figure 3: Logistic Regression Sigmoid Probability Curve
    # Compute the linear decision score z = X_scaled * coef + intercept
    z_test = log_reg.decision_function(X_test_scaled)
    sort_idx = np.argsort(z_test)
    z_sorted = z_test[sort_idx]
    prob_sorted = y_prob_lr[sort_idx]
    actual_sorted = y_test.iloc[sort_idx]

    plt.figure(figsize=(9, 6), dpi=300)
    # Continuous sigmoid curve
    z_continuous = np.linspace(-4, 4, 300)
    p_continuous = 1 / (1 + np.exp(-z_continuous))
    plt.plot(z_continuous, p_continuous, color="#059669", linewidth=2.8, label="Logistic Sigmoid Function: σ(z)")
    # Actual test points
    plt.scatter(
        z_test,
        y_test,
        c=[palette[val] for val in y_test],
        alpha=0.75,
        s=60,
        edgecolor="k",
        linewidth=0.5,
        label="Test Patients (Actual: 0=Healthy, 1=Disease)",
        zorder=5
    )
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Classification Threshold (p = 0.5)")
    plt.axvline(0.0, color="gray", linestyle=":", alpha=0.7, label="Decision Boundary (z = 0)")
    plt.title("Logistic Regression Sigmoid Curve (Log-Odds vs. Disease Likelihood)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Log-Odds Risk Score (z = β₀ + β₁X₁ + ... + βₙXₙ)", fontsize=12)
    plt.ylabel("Predicted Probability of Heart Disease: P(Y=1)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="center left", fontsize=10)
    plt.tight_layout()
    sigmoid_path = os.path.join(BASE_DIR, "logistic_regression_sigmoid.png")
    plt.savefig(sigmoid_path)
    plt.close()
    print(f"[OK] Saved sigmoid plot: {sigmoid_path}")

    # Figure 4: Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6), dpi=300)
    cm_labels = ["No Disease (0)", "Heart Disease (1)"]
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        annot_kws={"size": 18, "weight": "bold"}
    )
    plt.title("Confusion Matrix (Logistic Regression)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[OK] Saved confusion matrix: {cm_path}")

    # Figure 5: ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_lr, tpr_lr, color="#2563eb", linewidth=2.5, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
    plt.plot(fpr_rf, tpr_rf, color="#dc2626", linewidth=2.5, linestyle="--", label=f"Random Forest (AUC = {auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Guess (AUC = 0.500)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(BASE_DIR, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"[OK] Saved ROC curve: {roc_path}")

    # Figure 6: Feature Coefficients / Importance
    coefs = log_reg.coef_[0]
    feat_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": coefs})
    feat_df["AbsCoef"] = feat_df["Coefficient"].abs()
    feat_df = feat_df.sort_values(by="Coefficient", ascending=False)

    plt.figure(figsize=(9, 6), dpi=300)
    colors = ["#dc2626" if c > 0 else "#2563eb" for c in feat_df["Coefficient"]]
    sns.barplot(data=feat_df, x="Coefficient", y="Feature", palette=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Logistic Regression Feature Coefficients (Risk Factors)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Log-Odds Coefficient (Positive = Higher Risk, Negative = Lower Risk)", fontsize=12)
    plt.ylabel("Clinical Feature", fontsize=12)
    plt.tight_layout()
    feat_path = os.path.join(BASE_DIR, "feature_importance.png")
    plt.savefig(feat_path)
    plt.close()
    print(f"[OK] Saved feature importance plot: {feat_path}")

    # Save metrics JSON for the Web App
    metrics_summary = {
        "accuracy": f"{acc_lr * 100:.2f}%",
        "error_rate": f"{err_lr * 100:.2f}%",
        "precision": f"{prec_lr:.4f}",
        "recall": f"{rec_lr:.4f}",
        "specificity": f"{spec_lr:.4f}",
        "f1_score": f"{f1_lr:.4f}",
        "mcc": f"{mcc_lr:.4f}",
        "cohen_kappa": f"{kappa_lr:.4f}",
        "log_loss": f"{loss_lr:.4f}",
        "roc_auc": f"{auc_lr:.4f}",
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "rf_accuracy": f"{acc_rf * 100:.2f}%",
        "rf_auc": f"{auc_rf:.4f}",
    }

    metrics_json_path = os.path.join(BASE_DIR, "metrics_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[OK] Exported metrics summary JSON: {metrics_json_path}")
    print("\n[SUCCESS] Step 2 Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_pipeline()
