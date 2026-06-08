"""
Generate a realistic synthetic credit dataset (5,000 applicants).

Features are constructed with realistic correlations:
  - Higher loan-to-income → higher default probability
  - Longer employment, higher income → lower default probability
  - Previous defaults → strongly increases default probability
  - Credit utilization above 80% → increases default probability

Default rate: ~22%, consistent with subprime lending literature.
"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, N_SAMPLES, RANDOM_STATE

def generate(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── DEMOGRAPHICS ─────────────────────────────────────────────────
    age = rng.integers(20, 70, n)
    employment_type = rng.choice(
        ["Employed", "Self-employed", "Civil servant", "Unemployed"],
        p=[0.55, 0.20, 0.15, 0.10], size=n
    )
    housing_status = rng.choice(
        ["Owner", "Renter", "Living with family"],
        p=[0.40, 0.45, 0.15], size=n
    )

    # ── FINANCIALS ───────────────────────────────────────────────────
    # Income correlated with employment type
    base_income = np.where(
        employment_type == "Civil servant", 42_000,
        np.where(employment_type == "Employed", 38_000,
        np.where(employment_type == "Self-employed", 45_000, 18_000))
    )
    income_eur = np.clip(
        base_income + rng.normal(0, 10_000, n), 12_000, 200_000
    ).astype(int)

    employment_years = np.clip(
        rng.exponential(6, n) + np.where(employment_type == "Unemployed", -3, 0),
        0, 40
    ).round(1)

    loan_amount_eur = np.clip(
        rng.lognormal(9.5, 0.7, n), 1_000, 100_000
    ).astype(int)

    loan_duration_months = rng.choice([12, 24, 36, 48, 60, 84, 120], n)

    loan_purpose = rng.choice(
        ["Car", "Home improvement", "Debt consolidation",
         "Education", "Business", "Personal"],
        p=[0.25, 0.20, 0.20, 0.10, 0.15, 0.10], size=n
    )

    loan_to_income     = (loan_amount_eur / income_eur).round(4)
    existing_debts     = rng.integers(0, 8, n)
    credit_utilization_pct = np.clip(rng.beta(2, 4, n) * 100, 0, 100).round(1)
    num_previous_loans = rng.integers(0, 10, n)

    # Months since last default (999 = never defaulted)
    had_previous_default = rng.random(n) < 0.18
    months_since_last_default = np.where(
        had_previous_default,
        rng.integers(1, 120, n),
        999
    )

    # ── DEFAULT LABEL (logistic model) ───────────────────────────────
    log_odds = (
        -3.8
        + 1.8  * loan_to_income
        - 0.03 * employment_years
        - 0.01 * (income_eur / 10_000)
        + 0.15 * existing_debts
        + 0.02 * credit_utilization_pct
        + 1.20 * had_previous_default.astype(float)
        + 0.30 * (employment_type == "Unemployed").astype(float)
        - 0.20 * (housing_status == "Owner").astype(float)
        + 0.15 * (loan_purpose == "Debt consolidation").astype(float)
        + rng.logistic(0, 1, n)       # idiosyncratic noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.random(n) < prob_default).astype(int)

    df = pd.DataFrame({
        "age":                      age,
        "income_eur":               income_eur,
        "employment_years":         employment_years,
        "loan_amount_eur":          loan_amount_eur,
        "loan_duration_months":     loan_duration_months,
        "loan_to_income":           loan_to_income,
        "existing_debts":           existing_debts,
        "credit_utilization_pct":   credit_utilization_pct,
        "num_previous_loans":       num_previous_loans,
        "months_since_last_default":months_since_last_default,
        "loan_purpose":             loan_purpose,
        "housing_status":           housing_status,
        "employment_type":          employment_type,
        "default":                  default,
    })

    path = os.path.join(DATA_DIR, "credit_data.csv")
    df.to_csv(path, index=False)
    print(f"Dataset generated: {n} applicants, {default.mean()*100:.1f}% default rate")
    print(f"Saved to: {path}")
    return df

if __name__ == "__main__":
    generate()
