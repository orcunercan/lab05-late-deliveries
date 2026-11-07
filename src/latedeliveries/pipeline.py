"""Feature lists, the train/test split and the pipeline builder."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

SEED = 42
TEST_SIZE = 0.25
TARGET = "late"

# Hours with the most orders: lunch (12-13) and dinner (18-20).
RUSH_HOURS = [12, 13, 18, 19, 20]

# Only columns known at the moment an order is placed.
NUMERIC = ["distance_km", "items", "prep_minutes", "courier_months"]
CATEGORICAL = ["weather", "day_type", "vehicle", "rush"]

# Columns left out on purpose. The reasons are in the README.
EXCLUDED = ["order_id", "hour", "customer_rating"]


def add_rush(orders: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the orders with a rush column: "rush" or "normal"."""
    orders = orders.copy()
    is_rush = orders["hour"].isin(RUSH_HOURS)
    orders["rush"] = is_rush.map({True: "rush", False: "normal"})
    return orders


def load_split(orders: pd.DataFrame | None = None):
    """Return X_train, X_test, y_train, y_test: a stratified, seeded 75/25 split."""
    if orders is None:
        orders = make_orders()
    X = orders.drop(columns=[TARGET])
    y = orders[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def build_pipeline(model, numeric=None, categorical=None) -> Pipeline:
    """Return a pipeline: add the rush column, prepare the features, then the model."""
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    numbers = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    preprocess = ColumnTransformer(
        [
            ("num", numbers, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        [
            ("rush", FunctionTransformer(add_rush)),
            ("prep", preprocess),
            ("model", model),
        ]
    )
