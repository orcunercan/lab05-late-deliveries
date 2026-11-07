"""Shared fixtures: a small, seeded order table and a pipeline fitted on it."""

import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import TARGET, build_pipeline


@pytest.fixture(scope="session")
def small_orders():
    """400 invented orders, enough for fast tests."""
    return make_orders(n=400, seed=1)


@pytest.fixture(scope="session")
def fitted_pipeline(small_orders):
    """A logistic regression pipeline fitted on the small table."""
    X = small_orders.drop(columns=[TARGET])
    y = small_orders[TARGET]
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)


@pytest.fixture
def one_order(small_orders):
    """A single order as a one-row table, without the target."""
    return small_orders.drop(columns=[TARGET]).iloc[[0]].copy()