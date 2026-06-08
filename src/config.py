"""Configuration for the credit scoring model."""
import os

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
OUTPUT_TABLES = os.path.join(PROJECT_ROOT, "outputs", "tables")
OUTPUT_FIGS   = os.path.join(PROJECT_ROOT, "outputs", "figures")
for d in [DATA_DIR, OUTPUT_TABLES, OUTPUT_FIGS]:
    os.makedirs(d, exist_ok=True)

N_SAMPLES    = 5_000
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
THRESHOLD    = 0.50

NUMERIC_FEATURES = [
    "age", "income_eur", "employment_years",
    "loan_amount_eur", "loan_duration_months",
    "loan_to_income", "existing_debts",
    "credit_utilization_pct", "num_previous_loans",
    "months_since_last_default",
]

CATEGORICAL_FEATURES = [
    "loan_purpose", "housing_status", "employment_type",
]

TARGET = "default"
