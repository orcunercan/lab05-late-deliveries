# Late Deliveries

QuickBite can give a customer a $4 voucher when an order is placed. This project predicts, at that moment, the probability that an order will arrive late, and flags the orders above a threshold chosen to make the total cost as low as possible.

## Setup

```
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```
python -m latedeliveries.train
```

This one command recreates `reports/cv_results.csv`, `reports/metrics.json`, `models/late_orders.joblib` and `models/model_info.json`. The notebook `notebooks/analysis.ipynb` explains the decisions below with charts; it reads the test-set result that `train.py` saved and never scores the test set itself.

## Check it

```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns

The model may only use what is known **when the order is placed**. The feature lists are defined once, in `src/latedeliveries/pipeline.py`.

| Column | Used? | Reason |
|---|---|---|
| `distance_km`, `items`, `courier_months` | yes, numeric | Known when the order is placed |
| `prep_minutes` | yes, numeric | Treated as the kitchen's estimate, known when the order is accepted. Missing in 6% of orders; filled with the median |
| `weather`, `day_type`, `vehicle` | yes, category | Known when the order is placed |
| `rush` | yes, category | Made from `hour`: "rush" for 12, 13, 18, 19 and 20 (lunch and dinner), otherwise "normal". Built inside the pipeline, so the saved model takes a raw order |
| `customer_rating` | **no** | Given **after** delivery, so it is not known when the order is placed. It leaks the answer: in the training data every order rated 2 was late and no order rated 5 was |
| `order_id` | **no** | A label with no meaning; the model could only memorise it |
| `hour` | **no** | Replaced by `rush`, which scored better (see the experiments below) |

**Leakage check.** Adding `customer_rating` back to the chosen features raised cross-validated ROC-AUC from **0.816 to 0.920** and cut the lowest cost by about $1,000. A jump that large from a single column is a warning, not good news, so it stays out.

### Feature experiments: hour, rush and segments

Delivery work is not spread evenly over the day, so I tested whether rush hours, distance segments and order-size segments help. The business view in the training data supports all three ideas:

| Segment | Late rate | Segment | Late rate | Segment | Late rate |
|---|---|---|---|---|---|
| 0-2 km | 5.3% | 1-2 items | 11.4% | normal hour | 11.3% |
| 2-5 km | 11.3% | 3-5 items | 15.5% | rush hour | 23.2% |
| 5+ km | 33.7% | 6-8 items | 18.4% | | |

As model features, logistic regression, 5-fold cross-validation on the training data (also in `reports/feature_experiments.csv`):

| Variant | ROC-AUC (mean ± std) | Lowest CV cost |
|---|---|---|
| 1 hour as a number | 0.795 ± 0.034 | $4,031 |
| 2 hour as a category | 0.812 ± 0.034 | $4,096 |
| **3 rush only (chosen)** | **0.816 ± 0.034** | $4,042 |
| 4 rush + hour as a number | 0.815 ± 0.035 | $4,023 |
| 5 rush + hour as a category | 0.812 ± 0.034 | $4,089 |
| 6 rush + 3 distance and 3 size segments | 0.816 ± 0.033 | $4,032 |
| 7 segments replace raw distance and items | 0.798 ± 0.044 | $4,220 |
| Leak: chosen + `customer_rating` | 0.920 ± 0.014 | $3,035 |

- **Hour.** As one number, a linear model can only learn "later is riskier" and misses the lunch and dinner peaks. As a category it sees the peaks but must learn 13 separate hours from a few hundred orders each. Grouping the hours into rush and normal, from business knowledge, ranks orders best with fewer columns; adding the raw hour back adds nothing.
- **Distance and size segments** add nothing on top of the raw numbers and hurt when they replace them. Risk rises smoothly with distance and size, which the model already learns from the numbers, and banding throws detail away. Grouping helps when the relationship has peaks, as hour does, not when it is a straight rise.
- **Honest spread.** Apart from the leak, every difference is inside the fold-to-fold spread (about ±0.035), and the costs of variants 1, 3, 4 and 6 are within $20 on 2,250 orders. These are signals, not proof. I chose rush only as the simplest variant with the best ranking.

### Model choice

Five-fold stratified cross-validation on the training data only (`reports/cv_results.csv`). "Lowest CV cost" is the total cost on the 2,250 training orders at each model's own best threshold, from out-of-fold predictions.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best threshold | Lowest CV cost |
|---|---|---|---|---|---|---|---|
| Dummy baseline (most frequent) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.05 | $5,265 |
| Logistic regression | 0.872 ± 0.010 | 0.715 ± 0.081 | 0.290 ± 0.045 | 0.413 ± 0.058 | 0.816 ± 0.034 | 0.30 | **$4,042** |
| Logistic regression, `class_weight="balanced"` | 0.752 ± 0.013 | 0.355 ± 0.021 | 0.724 ± 0.051 | 0.476 ± 0.029 | 0.816 ± 0.035 | 0.70 | $4,044 |
| Random forest, balanced, 300 trees | 0.782 ± 0.014 | 0.376 ± 0.028 | 0.598 ± 0.051 | 0.461 ± 0.034 | 0.795 ± 0.037 | 0.60 | $4,180 |

- **Accuracy cannot choose.** The baseline flags nobody and is still 84% accurate, because 84% of orders are on time.
- **F1 alone would pick the balanced logistic regression.** The scenario's costs say otherwise: once each model gets its own threshold, the two logistic regressions cost the same ($4,042 against $4,044). `class_weight="balanced"` only shifts the probabilities upward, and choosing the threshold from costs undoes that shift.
- **The random forest costs $138 more** and ranks orders less well. The effects in this data add up one by one, which a linear model captures directly; a forest looks for combinations that are not there.
- I chose **logistic regression**: the lowest cost, the best ranking, and coefficients that a manager can read. The grid search below decides its class weight.

### Tuning

`GridSearchCV` on the logistic regression pipeline, 5-fold stratified, over `model__C` = 0.01, 0.1, 1, 10 and `model__class_weight` = None, balanced, {0: 1, 1: 3}: **12 combinations**.

- **Scoring: ROC-AUC.** The threshold is chosen afterwards from costs, so what tuning needs to improve is how well the model **ranks** late orders above on-time ones, which is what ROC-AUC measures, independent of any threshold.
- **Best settings:** `C=0.1`, `class_weight=None`, ROC-AUC **0.817 ± 0.032**, against **0.816 ± 0.035** untuned.
- **Plainly: tuning did not beat the untuned model by more than the fold-to-fold spread.** A gain of 0.001 is far inside ±0.03. The tuned model is kept because it is no worse, not because it is better.

### Threshold

The model gives each order a probability; an order is flagged when its probability is at least the threshold. The cost rule is

    total cost = $4 × flagged orders + $15 × late orders that were not flagged

and it is written once, as `order_cost` in `src/latedeliveries/evaluation.py`.

A default threshold of 0.5 would only be right if a missed late order and a wasted voucher cost the same. Here a miss costs 3.75 times a voucher, so it pays to flag an order well before it is more likely late than not. I took out-of-fold probabilities for the training data with `cross_val_predict`, priced every threshold from 0.05 to 0.95 in steps of 0.05, and kept the cheapest: **0.30** (cost $3,990 on the 2,250 training orders, against $5,265 for flagging nobody and $9,000 for flagging everybody). The test set played no part in this choice. The full cost curve is in `models/model_info.json` and charted in the notebook.

## Result

The final pipeline was fitted on the whole training set and scored **once** on the 750 test orders, at threshold 0.30 (`reports/metrics.json`).

|  | Predicted on time | Predicted late |
|---|---|---|
| **Actually on time** | 590 | 43 |
| **Actually late** | 63 | 54 |

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.557 | 0.462 | 0.505 | 0.822 |

| Policy | Total cost on 750 test orders |
|---|---|
| Flag nobody (today) | $1,755 |
| Flag everybody | $3,000 |
| **Our model, threshold 0.30** | **$1,333** |

The model saves **$422 (24%)** against today's policy. In business terms: of 117 late orders it caught 54 in advance, each costing a $4 voucher instead of $15, and it gave 43 on-time customers a voucher they did not need.

How far I trust it: the test ROC-AUC (0.822) matches cross-validation (0.816 ± 0.034), so the model behaves the same on unseen orders, but 117 late orders is a small number, and a saving of a few hundred dollars on 750 orders could move by roughly a hundred either way on another sample. It is a clear improvement on flagging nobody, not a precise figure.

## What the tests cover

`python -m pytest` runs 18 tests in under a second, with shared fixtures in `tests/conftest.py`:

- **Feature list:** the left-out columns (`customer_rating`, `order_id`, `hour`, and the target) are not features (parametrized).
- **Split:** stratified and repeatable with the fixed seed.
- **Rush:** lunch and dinner hours become "rush", others "normal" (parametrized).
- **Pipeline:** probabilities are between 0 and 1 and sum to 1; an order with a missing `prep_minutes` and an unseen weather value ("hail") still gets a prediction.
- **Cost and threshold:** hand-worked examples, including one where flagging nobody is cheapest and one where flagging everybody is, and a five-step example of the threshold search whose costs are written out in the test.
- **Saved model:** a pipeline saved with joblib to `tmp_path` and loaded back gives exactly the same probabilities.

**Not tested:** the full `train.py` run. It takes several seconds and its numbers are checked by running it and comparing against the committed `reports/metrics.json`; a test that retrains every model would make the suite slow without protecting any logic the smaller tests miss.

## Known limitations

- The data is invented, and its effects add up neatly, which favours a linear model. Real deliveries have interactions (a far, large order in a rush hour is likely worse than the sum of its parts), where a tree model could win.
- The table has no location. City area, restaurant and live traffic would very likely matter more than any column here.
- `prep_minutes` is treated as known when the order is placed. If the kitchen time is only known afterwards, it should be replaced by an estimate or dropped.
- The rush hours are fixed by hand. If demand patterns change, the list needs updating.
- The costs ($4 and $15) are assumed fixed. If the cost of a missed late order changes, the threshold must be chosen again; the cost curve in `model_info.json` makes that quick.
- The saving should be confirmed in a small live trial before the voucher is offered to every customer.
