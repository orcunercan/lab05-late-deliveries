"""A saved and reloaded pipeline must predict exactly as before."""

import joblib
import numpy as np


def test_saved_model_predicts_the_same_after_reloading(
    fitted_pipeline, small_orders, tmp_path
):
    X = small_orders.drop(columns="late")
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    reloaded = joblib.load(path)
    assert np.array_equal(fitted_pipeline.predict_proba(X), reloaded.predict_proba(X))
