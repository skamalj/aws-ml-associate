"""Feature engineering for the SafeDrive fraud model (taught step by step in notebook 1.2a)."""
import numpy as np
import pandas as pd

LABEL = "is_fraud"
CLAIM_TYPES = ["collision", "theft", "glass", "fire", "third_party"]
CHANNELS = ["app", "phone", "agent"]
REGIONS = ["North", "South", "East", "West"]

FEATURES = (
    ["claim_amount_log", "claim_to_value_ratio", "claim_to_premium_ratio", "report_delay_days",
     "night_incident", "no_police_report", "witnesses", "num_vehicles", "age", "young_driver",
     "tenure_months", "new_customer", "prior_claims", "vehicle_age", "annual_premium"]
    + [f"claim_type_{c}" for c in CLAIM_TYPES]
    + [f"channel_{c}" for c in CHANNELS]
    + [f"region_{r}" for r in REGIONS]
)


def clean_claims(claims):
    """Fix the quality problems planted in claims_v2 (explained in notebook 1.3)."""
    df = claims.drop_duplicates(subset="claim_id").copy()
    df["claim_type"] = df["claim_type"].str.strip().str.lower()
    df.loc[df["report_delay_days"] < 0, "report_delay_days"] = 0
    cap = df["claim_amount"].quantile(0.99)
    df["claim_amount"] = df["claim_amount"].clip(upper=cap).fillna(df["claim_amount"].median())
    df["incident_hour"] = df["incident_hour"].fillna(12)
    df["police_report"] = df["police_report"].fillna("N")
    return df


def build_features(claims, policies):
    """Join claims with policies and return a numeric frame: FEATURES (+ LABEL if present)."""
    df = clean_claims(claims).merge(policies, on="policy_id", how="left")
    out = pd.DataFrame(index=df.index)
    out["claim_amount_log"] = np.log1p(df["claim_amount"])
    out["claim_to_value_ratio"] = df["claim_amount"] / df["vehicle_value"]
    out["claim_to_premium_ratio"] = df["claim_amount"] / df["annual_premium"]
    out["report_delay_days"] = df["report_delay_days"]
    out["night_incident"] = ((df["incident_hour"] < 5) | (df["incident_hour"] > 22)).astype(int)
    out["no_police_report"] = (df["police_report"] == "N").astype(int)
    out["witnesses"] = df["witnesses"]
    out["num_vehicles"] = df["num_vehicles"]
    out["age"] = df["age"]
    out["young_driver"] = (df["age"] < 25).astype(int)
    out["tenure_months"] = df["tenure_months"]
    out["new_customer"] = (df["tenure_months"] < 6).astype(int)
    out["prior_claims"] = df["prior_claims"]
    out["vehicle_age"] = pd.to_datetime(df["claim_date"]).dt.year - df["vehicle_year"]
    out["annual_premium"] = df["annual_premium"]
    for c in CLAIM_TYPES:
        out[f"claim_type_{c}"] = (df["claim_type"] == c).astype(int)
    for c in CHANNELS:
        out[f"channel_{c}"] = (df["channel"] == c).astype(int)
    for r in REGIONS:
        out[f"region_{r}"] = (df["region"] == r).astype(int)
    out = out[FEATURES].astype(float).round(5)
    if LABEL in df:
        out.insert(0, LABEL, df[LABEL].astype(int).values)
    out.insert(0, "claim_id", df["claim_id"].values)
    return out
