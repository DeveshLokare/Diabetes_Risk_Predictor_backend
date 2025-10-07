import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, conint
from typing import Optional, List
import shap
import math


MODEL_PATH = os.getenv("MODEL_PATH", "xgb_diabetes_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "")
BACKGROUND_PATH = os.getenv("BACKGROUND_PATH", "background_sample.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

FEATURE_COLUMNS = [
    'Age','Polyuria','Polydipsia','sudden_weight_loss','weakness','Polyphagia',
    'Genital_thrush','visual_blurring','Itching','Irritability',
    'delayed_healing','partial_paresis','muscle_stiffness','Alopecia','Obesity'
]

app = Flask(__name__)
CORS(app)


model = joblib.load(MODEL_PATH)
scaler = None
if SCALER_PATH:
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None

background = None
if os.path.exists(BACKGROUND_PATH):
    try:
        background = joblib.load(BACKGROUND_PATH)
    except Exception:
        background = None


try:
    if background is not None:
        explainer = shap.TreeExplainer(model, data=background)
    else:
        explainer = shap.TreeExplainer(model)
except Exception:
    try:
        explainer = shap.Explainer(model)
    except Exception:
        explainer = None


class InputSchema(BaseModel):
    Age: conint(ge=0, le=120)
    Polyuria: conint(ge=0, le=1)
    Polydipsia: conint(ge=0, le=1)
    sudden_weight_loss: conint(ge=0, le=1)
    weakness: conint(ge=0, le=1)
    Polyphagia: conint(ge=0, le=1)
    Genital_thrush: conint(ge=0, le=1)
    visual_blurring: conint(ge=0, le=1)
    Itching: conint(ge=0, le=1)
    Irritability: conint(ge=0, le=1)
    delayed_healing: conint(ge=0, le=1)
    partial_paresis: conint(ge=0, le=1)
    muscle_stiffness: conint(ge=0, le=1)
    Alopecia: conint(ge=0, le=1)
    Obesity: conint(ge=0, le=1)
    threshold: Optional[float] = None

def preprocess_input(data: InputSchema) -> pd.DataFrame:
    row = {col: getattr(data, col) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    if scaler is not None:
        try:
            df[["Age"]] = scaler.transform(df[["Age"]])
        except Exception:
            pass
    return df

def _get_shap_values_for_class1(X: pd.DataFrame):
    """Return shap values array for class 1 (shape: n_samples x n_features)."""
    if explainer is None:
        return None

    # Common case: TreeExplainer.shap_values returns a list (one per class) for classifiers
    try:
        sv = explainer.shap_values(X)
        if isinstance(sv, list) and len(sv) >= 2:
            # choose class 1
            arr = np.array(sv[1])
        else:
            arr = np.array(sv)
        # ensure shape is (n_samples, n_features)
        if arr.ndim == 3:
            # arr[class_index, n_samples, n_features] -> pick last class if shape[0] matches classes
            arr = arr[-1]
        return arr
    except Exception:
        # Newer SHAP API: call explainer(X) -> object with .values
        try:
            res = explainer(X)
            vals = getattr(res, "values", None)
            if vals is None:
                return None
            vals = np.array(vals)
            # if multi-dim, choose class 1 (index -1 if two-class)
            if vals.ndim == 3:
                # shape (classes, n_samples, n_features)
                return vals[-1]
            elif vals.ndim == 2:
                # shape (n_samples, n_features)
                return vals
            else:
                return None
        except Exception:
            return None

def generate_text_explanation(shap_vals_row: np.ndarray, feature_names: List[str], feature_values: List):
    """
    Build human readable explanation sentences describing top contributors.
    - shap_vals_row: shape (n_features,)
    - feature_names: list of feature names
    - feature_values: corresponding feature values (0/1 or numeric)
    """
    if shap_vals_row is None:
        return "SHAP explanations are unavailable on the server."

    abs_vals = np.abs(shap_vals_row)
    total_abs = abs_vals.sum()
    if total_abs == 0 or math.isclose(total_abs, 0.0):
        return "Model indicates no strong feature-level contributions for this sample."

    # Combine and sort top contributors
    feats = list(zip(feature_names, feature_values, shap_vals_row, abs_vals))
    feats_sorted = sorted(feats, key=lambda x: x[3], reverse=True)

    # Top 3 contributors (positive = increases risk, negative = decreases)
    top_k = feats_sorted[:3]
    sentences = []
    for name, val, shap_val, absval in top_k:
        pct = (absval / total_abs) * 100
        direction = "increases" if shap_val > 0 else "decreases"
        # improve readability for binary features
        val_str = str(val)
        sentences.append(f"{name} = {val_str} {direction} predicted diabetes risk (contribution ~ {pct:.1f}% of the model's feature-based explanation).")

    # Also mention any large negative contributor
    # find the strongest negative (if not in top_k)
    negs = [f for f in feats_sorted if f[2] < 0]
    if negs and negs[0] not in top_k:
        name, val, shap_val, absval = negs[0]
        pct = (absval / total_abs) * 100
        sentences.append(f"Notably, {name} = {val} decreases predicted risk (~{pct:.1f}% contribution).")

    return " ".join(sentences)

# ROUTES
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400
        data = InputSchema(**payload)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 422
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    X = preprocess_input(data)
    prob = float(model.predict_proba(X)[:, 1][0])
    threshold = data.threshold if data.threshold is not None else THRESHOLD
    label = "Yes" if prob >= threshold else "No"

    # SHAP explanation
    shap_arr = _get_shap_values_for_class1(X)
    shap_row = shap_arr[0] if (shap_arr is not None and shap_arr.shape[0] >= 1) else None
    expl_text = generate_text_explanation(shap_row, FEATURE_COLUMNS, list(X.iloc[0].values))
    # also prepare a small JSON-friendly list of top feature contributions
    shap_details = None
    if shap_row is not None:
        contributions = []
        for feat, val, s in zip(FEATURE_COLUMNS, X.iloc[0].values, shap_row):
            contributions.append({"feature": feat, "value": float(val), "shap_value": float(s), "abs": abs(float(s))})
        contributions = sorted(contributions, key=lambda x: x["abs"], reverse=True)[:10]
        shap_details = contributions

    response = {
        "probability": prob,
        "threshold": threshold,
        "label": label,
        "explanation_text": expl_text,
        "shap_details": shap_details
    }
    return jsonify(response), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)