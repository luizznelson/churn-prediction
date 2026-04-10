import pandas as pd
from pathlib import Path

RAW = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT = Path("data/processed/telco_clean.csv")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW)

    # TotalCharges has ~11 blank strings — coerce to float, fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Target: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop non-feature identifier
    df = df.drop(columns=["customerID"])

    # Binary Yes/No columns (and service-level "No X service" variants) → 1/0
    binary_map = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
    binary_cols = [
        col for col in df.columns
        if df[col].dtype == object and set(df[col].dropna().unique()).issubset(binary_map.keys())
        and col not in ("Contract", "PaymentMethod", "InternetService")
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    df.to_csv(OUT, index=False)

    churn_rate = df["Churn"].mean() * 100
    print(f"Saved: {OUT}")
    print(f"Rows:  {len(df):,}")
    print(f"Churn: {churn_rate:.2f}%")

if __name__ == "__main__":
    main()
