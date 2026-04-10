import json
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

DATA = Path("data/processed/telco_clean.csv")
MODEL_OUT = Path("models/modelo_churn.joblib")
OUTPUTS = Path("outputs")


def main():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    pipeline = Pipeline([
        ("prep", ct),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_roc":      round(roc_auc_score(y_test, y_proba), 4),
        "precision":    round(precision_score(y_test, y_pred), 4),
        "recall":       round(recall_score(y_test, y_pred), 4),
        "f1":           round(f1_score(y_test, y_pred), 4),
        "accuracy":     round(accuracy_score(y_test, y_pred), 4),
        "test_size":    len(y_test),
        "churn_in_test": int(y_test.sum()),
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Feature importance — OHE expands categorical cols; get full name list
    feature_names = list(numeric_cols) + list(
        pipeline["prep"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
    )
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": pipeline["clf"].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(OUTPUTS / "feature_importance.csv", index=False)

    # Segment CSVs computed on the full dataset
    def segment(group_col, out_file):
        agg = (
            df.groupby(group_col)["Churn"]
            .agg(churn_rate="mean", total="count")
            .reset_index()
        )
        agg["churn_rate"] = agg["churn_rate"].round(4)
        agg.to_csv(OUTPUTS / out_file, index=False)

    segment("Contract", "churn_por_contrato.csv")
    segment("PaymentMethod", "churn_por_pagamento.csv")

    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "13-24", "25-48", "49-72"]
    df["faixa_tempo"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)
    faixa = (
        df.groupby("faixa_tempo", observed=True)["Churn"]
        .agg(churn_rate="mean", total="count")
        .reset_index()
    )
    faixa["churn_rate"] = faixa["churn_rate"].round(4)
    faixa.to_csv(OUTPUTS / "churn_por_faixa_tempo.csv", index=False)

    joblib.dump(pipeline, MODEL_OUT)

    print(f"AUC-ROC:   {metrics['auc_roc']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1:        {metrics['f1']}")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Test size: {metrics['test_size']} rows  |  Churn in test: {metrics['churn_in_test']}")
    print(f"Model:     {MODEL_OUT}")
    print(f"Outputs:   {OUTPUTS}/")


if __name__ == "__main__":
    main()
