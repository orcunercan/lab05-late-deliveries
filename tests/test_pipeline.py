"""Tests for the feature lists, the split and the pipeline."""

import numpy as np
import pandas as pd
import pytest

from latedeliveries.pipeline import (
    CATEGORICAL,
    EXCLUDED,
    NUMERIC,
    add_rush,
    load_split,
)


@pytest.mark.parametrize("column", ["customer_rating", "order_id", "hour", "late"])
def test_left_out_columns_are_not_features(column):
    assert column not in NUMERIC + CATEGORICAL


def test_excluded_list_names_the_leaky_column():
    assert "customer_rating" in EXCLUDED


def test_split_is_stratified_and_repeatable():
    X_train, X_test, y_train, y_test = load_split()
    X_train_again, _, _, _ = load_split()
    assert list(X_train.index) == list(X_train_again.index)
    assert len(X_test) == 750
    assert abs(y_train.mean() - y_test.mean()) < 0.01


@pytest.mark.parametrize(
    ("hour", "expected"), [(12, "rush"), (19, "rush"), (10, "normal"), (22, "normal")]
)
def test_rush_marks_lunch_and_dinner_hours(hour, expected):
    orders = pd.DataFrame({"hour": [hour]})
    assert add_rush(orders)["rush"].iloc[0] == expected


def test_probabilities_are_valid(fitted_pipeline, small_orders):
    probabilities = fitted_pipeline.predict_proba(small_orders.drop(columns="late"))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_missing_prep_time_and_unseen_weather_still_predict(fitted_pipeline, one_order):
    one_order["prep_minutes"] = np.nan
    one_order["weather"] = "hail"
    probability = fitted_pipeline.predict_proba(one_order)[0, 1]
    assert 0 <= probability <= 1
