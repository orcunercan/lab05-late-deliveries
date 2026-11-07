"""Train, compare, tune, choose a threshold, test once and save: one command.

Run it with: python -m latedeliveries.train
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    policy_costs,
    score_at_threshold,
)
from latedeliveries.pipeline import (
    CATEGORICAL,
    EXCLUDED,
    NUMERIC,
    SEED,
    build_pipeline,
    load_split,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# The baseline never flags an order, so its precision is 0 by definition.
SCORERS = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
CHOSEN_MODEL = "logreg_balanced"
PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced", {0: 1, 1: 3}],
}


def candidate_models() -> dict:
    """The baseline and the three real candidates, all with fixed seeds."""
    return {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
        "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
        ),
    }


def make_folds() -> StratifiedKFold:
    """Five stratified folds, shuffled with the fixed seed."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def compare_models(X_train, y_train) -> pd.DataFrame:
    """Cross-validate every candidate on the training data only."""
    rows = []
    for name, model in candidate_models().items():
        pipe = build_pipeline(model)
        scores = cross_validate(
            pipe, X_train, y_train, cv=make_folds(), scoring=SCORERS
        )
        row = {"model": name}
        for metric in METRICS:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 3)
            row[f"{metric}_std"] = round(float(values.std()), 3)
        probabilities = cross_val_predict(
            pipe, X_train, y_train, cv=make_folds(), method="predict_proba"
        )[:, 1]
        best = choose_threshold(y_train, probabilities)
        row["best_threshold"] = best
        row["lowest_cv_cost"] = cost_by_threshold(y_train, probabilities)[best]
        rows.append(row)
    return pd.DataFrame(rows)


def tune(X_train, y_train) -> GridSearchCV:
    """Grid search the chosen model, scored by ROC-AUC."""
    search = GridSearchCV(
        build_pipeline(candidate_models()[CHOSEN_MODEL]),
        PARAM_GRID,
        cv=make_folds(),
        scoring="roc_auc",
    )
    search.fit(X_train, y_train)
    return search


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    # Steps 1-4 use the training data only.
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    print(comparison.to_string(index=False))

    search = tune(X_train, y_train)
    untuned = comparison.set_index("model").loc[CHOSEN_MODEL]
    tuning = {
        "scoring": "roc_auc",
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_roc_auc": round(float(search.best_score_), 3),
        "best_cv_roc_auc_std": round(
            float(search.cv_results_["std_test_score"][search.best_index_]), 3
        ),
        "untuned_cv_roc_auc": float(untuned["roc_auc_mean"]),
        "untuned_cv_roc_auc_std": float(untuned["roc_auc_std"]),
    }
    print("Tuning:", tuning)

    final = search.best_estimator_
    oof = cross_val_predict(
        final, X_train, y_train, cv=make_folds(), method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, oof)
    cv_costs = cost_by_threshold(y_train, oof)
    print("Threshold chosen on the training data:", threshold)

    # Step 5: the test set, used once.
    test_probabilities = final.predict_proba(X_test)[:, 1]
    metrics = score_at_threshold(y_test, test_probabilities, threshold)
    metrics["policy_costs"] = policy_costs(y_test, test_probabilities, threshold)
    metrics["test_orders"] = len(y_test)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Test set:", metrics)

    # Step 6: save the pipeline and what is needed to trust it.
    joblib.dump(final, MODELS_DIR / "late_orders.joblib")
    info = {
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": threshold,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "excluded_columns": EXCLUDED,
        "model": CHOSEN_MODEL,
        "tuning": tuning,
        "cv_cost_by_threshold": {str(t): c for t, c in cv_costs.items()},
        "test_results": metrics,
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved {MODELS_DIR / 'late_orders.joblib'} and model_info.json")


if __name__ == "__main__":
    main()
