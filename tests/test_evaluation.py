"""Tests for the cost rule and the threshold choice, on examples worked out by hand."""

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
)


def test_cost_of_a_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two vouchers (2 x 4) and one missed late order (1 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


def test_flagging_nobody_is_cheapest_when_no_order_is_late():
    y_true = [0, 0, 0]
    probabilities = [0.1, 0.2, 0.3]
    threshold = choose_threshold(y_true, probabilities)
    # every order stays below the threshold, so nobody is flagged and it costs $0
    assert all(p < threshold for p in probabilities)
    assert cost_by_threshold(y_true, probabilities)[threshold] == 0


def test_flagging_everybody_is_cheapest_when_every_order_is_late():
    y_true = [1, 1, 1]
    probabilities = [0.1, 0.2, 0.3]
    # flag all three: 3 x 4 = 12, far less than missing them: 3 x 15 = 45
    assert choose_threshold(y_true, probabilities) == 0.05


def test_threshold_on_a_hand_worked_example():
    y_true = [1, 0, 0, 1]
    probabilities = [0.9, 0.6, 0.2, 0.4]
    # 0.05 to 0.20: flag all four                    -> 4 x 4 = 16
    # 0.25 to 0.40: flag 0.9, 0.6, 0.4, miss none    -> 3 x 4 = 12 (lowest)
    # 0.45 to 0.60: flag 0.9, 0.6, miss the 0.4 late -> 8 + 15 = 23
    # 0.65 to 0.90: flag 0.9, miss the 0.4 late      -> 4 + 15 = 19
    # 0.95:         flag none, miss both late        -> 2 x 15 = 30
    assert choose_threshold(y_true, probabilities) == 0.25
    assert cost_by_threshold(y_true, probabilities)[0.25] == 12


def test_policy_costs_on_a_hand_worked_example():
    y_true = [1, 0, 0, 1]
    probabilities = [0.9, 0.6, 0.2, 0.4]
    costs = policy_costs(y_true, probabilities, threshold=0.5)
    assert costs == {"flag_nobody": 30, "flag_everybody": 16, "model": 23}
