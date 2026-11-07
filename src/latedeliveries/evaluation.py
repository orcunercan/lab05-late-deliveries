"""The cost rule, the threshold choice and the test-set metrics."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER_COST = 4  # every flagged order gets a $4 voucher
MISSED_LATE_COST = 15  # a late order that was not flagged costs $15
THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]  # 0.05 to 0.95


def order_cost(y_true, flagged) -> int:
    """Total cost: $4 per flagged order plus $15 per late order not flagged."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)
    vouchers = int(flagged.sum())
    missed_late = int(((y_true == 1) & ~flagged).sum())
    return VOUCHER_COST * vouchers + MISSED_LATE_COST * missed_late


def cost_by_threshold(y_true, probabilities, thresholds=THRESHOLDS) -> dict:
    """Return {threshold: total cost} when every order at or above it is flagged."""
    probabilities = np.asarray(probabilities)
    return {t: order_cost(y_true, probabilities >= t) for t in thresholds}


def choose_threshold(y_true, probabilities, thresholds=THRESHOLDS) -> float:
    """Return the threshold with the lowest total cost (the lowest one on a tie)."""
    costs = cost_by_threshold(y_true, probabilities, thresholds)
    return min(costs, key=lambda t: (costs[t], t))


def policy_costs(y_true, probabilities, threshold: float) -> dict:
    """Cost of flagging nobody, flagging everybody, and using the model."""
    y_true = np.asarray(y_true)
    nobody = np.zeros(len(y_true), dtype=bool)
    everybody = np.ones(len(y_true), dtype=bool)
    model = np.asarray(probabilities) >= threshold
    return {
        "flag_nobody": order_cost(y_true, nobody),
        "flag_everybody": order_cost(y_true, everybody),
        "model": order_cost(y_true, model),
    }


def score_at_threshold(y_true, probabilities, threshold: float) -> dict:
    """Confusion matrix and metrics at the chosen threshold."""
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": round(float(precision_score(y_true, predicted)), 3),
        "recall": round(float(recall_score(y_true, predicted)), 3),
        "f1": round(float(f1_score(y_true, predicted)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 3),
    }
