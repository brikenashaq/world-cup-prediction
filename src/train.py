"""
src/train.py  —  Train XGBoost, calibrate, save model
Run: python src/train.py
Out: models/model.pkl  models/label_encoder.pkl  models/metadata.json
"""
import json, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
MDIR = ROOT / "models"
MDIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "elo_diff",
    "home_form","away_form",
    "home_avg_goals","away_avg_goals",
    "home_avg_conceded","away_avg_conceded",
    "neutral",
]

def main():
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=FEATURE_COLS).sort_values("date").reset_index(drop=True)
    print(f"Total rows: {len(df)}")

    X = df[FEATURE_COLS].copy()
    X["neutral"] = X["neutral"].astype(int)

    le = LabelEncoder()
    y  = le.fit_transform(df["result"])
    print(f"Classes: {le.classes_}")

    # Time-based split — no shuffle, respect chronological order
    split    = int(len(df) * 0.80)
    X_train  = X.iloc[:split];  X_test  = X.iloc[split:]
    y_train  = y[:split];       y_test  = y[split:]
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    weights = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators     = 600,
        max_depth        = 3,
        learning_rate    = 0.02,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 10,
        eval_metric      = "mlogloss",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_train, y_train,
              sample_weight=weights,
              eval_set=[(X_test, y_test)],
              verbose=100)

    # Calibrate — makes probabilities reliable for simulator
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:
        cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    cal.fit(X_test, y_test)

    y_pred = cal.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("=== Feature Importance ===")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(imp.to_string())

    joblib.dump(cal, MDIR / "model.pkl")
    joblib.dump(le,  MDIR / "label_encoder.pkl")
    with open(MDIR / "metadata.json","w") as f:
        json.dump({
            "feature_cols": FEATURE_COLS,
            "classes":      list(le.classes_),
            "label_map":    {int(i): c for i,c in enumerate(le.classes_)},
        }, f, indent=2)

    print("\n✅ Model saved to models/")

if __name__ == "__main__":
    main()