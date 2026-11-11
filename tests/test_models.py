"""Pipeline behaviour: split, fit, evaluate, feature importance."""
import pytest

from config import TARGET, TEST_SIZE
from models import build_pipelines, evaluate, get_feature_importance, split_data


@pytest.fixture(scope="module")
def split(credit_df):
    return split_data(credit_df)


def test_split_is_stratified(credit_df, split):
    X_train, X_test, y_train, y_test = split
    assert len(X_test) == pytest.approx(len(credit_df) * TEST_SIZE, abs=1)
    overall = credit_df[TARGET].mean()
    assert abs(y_train.mean() - overall) < 0.01
    assert abs(y_test.mean() - overall) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)


def test_target_not_in_features(split):
    X_train, *_ = split
    assert TARGET not in X_train.columns


def test_three_named_pipelines():
    pipes = build_pipelines()
    assert list(pipes) == ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    for pipe in pipes.values():
        assert list(pipe.named_steps) == ["preprocessor", "classifier"]


def test_logistic_regression_beats_random(split):
    X_train, X_test, y_train, y_test = split
    model = build_pipelines()["Logistic Regression"].fit(X_train, y_train)
    res = evaluate(model, X_test, y_test, "Logistic Regression")
    assert res["roc_auc"] > 0.60
    assert 0 <= res["brier_score"] <= 0.25
    assert res["y_proba"].min() >= 0 and res["y_proba"].max() <= 1


def test_unseen_category_does_not_crash(split):
    """OneHotEncoder(handle_unknown='ignore') must tolerate new categories at scoring time."""
    X_train, X_test, y_train, _ = split
    model = build_pipelines()["Logistic Regression"].fit(X_train, y_train)
    X_new = X_test.head(5).copy()
    X_new["loan_purpose"] = "Wedding"
    proba = model.predict_proba(X_new)[:, 1]
    assert proba.shape == (5,)


def test_feature_importance_covers_all_encoded_features(split):
    X_train, _, y_train, _ = split
    model = build_pipelines()["Logistic Regression"].fit(X_train, y_train)
    fi = get_feature_importance(model, list(X_train.columns), "Logistic Regression")
    assert len(fi) == model.named_steps["preprocessor"].transform(X_train.head(1)).shape[1]
    assert fi["Importance"].is_monotonic_decreasing
