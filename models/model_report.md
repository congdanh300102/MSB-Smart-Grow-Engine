# Product Propensity - Logistic Regression report

- model_id: `sge_propensity_logreg`  ·  version: `v3.0`  ·  trained: 2026-09-04 14:21:28
- features (doc a.): x1_monthly_spending, x2_income, x3_digital_activity, x4_salary_account, x5_campaign_response, x6_product_gap
- 1 model / target product  ·  75/25 stratified split  ·  `class_weight=balanced`  ·  5-fold CV

## Test-set metrics

| product | n_test | pos_rate | ROC-AUC | PR-AUC | LogLoss | Brier | Acc | Precision | Recall | F1 | KS | CV-AUC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CREDIT_CARD (P001) | 3785 | 0.325 | 0.708 | 0.564 | 0.619 | 0.215 | 0.662 | 0.485 | 0.630 | 0.548 | 0.311 | 0.725 |
| LOAN (P004) | 4440 | 0.266 | 0.687 | 0.436 | 0.639 | 0.225 | 0.622 | 0.378 | 0.653 | 0.479 | 0.274 | 0.693 |
| DEPOSIT (P007) | 3856 | 0.233 | 0.705 | 0.434 | 0.624 | 0.217 | 0.658 | 0.365 | 0.638 | 0.465 | 0.309 | 0.698 |
| INVESTMENT (P009) | 5222 | 0.259 | 0.759 | 0.550 | 0.582 | 0.197 | 0.703 | 0.451 | 0.673 | 0.540 | 0.394 | 0.760 |

## Learned coefficients  (Z = b + Σ wᵢ·Xᵢ, on original 0–1 feature scale)

### CREDIT_CARD (P001)

| term | coefficient | odds ratio |
|---|--:|--:|
| intercept | -2.6742 | 0.069 |
| x1_monthly_spending | +1.2681 | 3.554 |
| x2_income | +0.3747 | 1.455 |
| x3_digital_activity | +0.8117 | 2.252 |
| x4_salary_account | +0.5834 | 1.792 |
| x5_campaign_response | +1.6969 | 5.457 |
| x6_product_gap | +0.9787 | 2.661 |

Confusion matrix (test, threshold 0.5): TN=1733 FP=823 FN=455 TP=774

### LOAN (P004)

| term | coefficient | odds ratio |
|---|--:|--:|
| intercept | -2.6417 | 0.071 |
| x1_monthly_spending | +0.8303 | 2.294 |
| x2_income | +0.8641 | 2.373 |
| x3_digital_activity | +0.7014 | 2.017 |
| x4_salary_account | +0.5575 | 1.746 |
| x5_campaign_response | +2.0763 | 7.975 |
| x6_product_gap | +1.0458 | 2.846 |

Confusion matrix (test, threshold 0.5): TN=1990 FP=1269 FN=410 TP=771

### DEPOSIT (P007)

| term | coefficient | odds ratio |
|---|--:|--:|
| intercept | -2.9886 | 0.050 |
| x1_monthly_spending | +1.1961 | 3.307 |
| x2_income | +0.7263 | 2.068 |
| x3_digital_activity | +0.5104 | 1.666 |
| x4_salary_account | +0.0415 | 1.042 |
| x5_campaign_response | +1.7097 | 5.528 |
| x6_product_gap | +2.3914 | 10.928 |

Confusion matrix (test, threshold 0.5): TN=1963 FP=995 FN=325 TP=573

### INVESTMENT (P009)

| term | coefficient | odds ratio |
|---|--:|--:|
| intercept | -3.1997 | 0.041 |
| x1_monthly_spending | +1.1650 | 3.206 |
| x2_income | +0.8566 | 2.355 |
| x3_digital_activity | +0.6769 | 1.968 |
| x4_salary_account | +0.3206 | 1.378 |
| x5_campaign_response | +2.4858 | 12.011 |
| x6_product_gap | +2.0711 | 7.934 |

Confusion matrix (test, threshold 0.5): TN=2758 FP=1109 FN=443 TP=912

## Calibration (test, deciles of predicted P)

### CREDIT_CARD

| decile | n | mean predicted P | observed rate |
|--:|--:|--:|--:|
| 1 | 379 | 0.172 | 0.121 |
| 2 | 378 | 0.252 | 0.159 |
| 3 | 379 | 0.315 | 0.211 |
| 4 | 378 | 0.374 | 0.235 |
| 5 | 379 | 0.428 | 0.237 |
| 6 | 378 | 0.483 | 0.304 |
| 7 | 378 | 0.546 | 0.381 |
| 8 | 379 | 0.617 | 0.425 |
| 9 | 378 | 0.699 | 0.503 |
| 10 | 379 | 0.833 | 0.670 |

### LOAN

| decile | n | mean predicted P | observed rate |
|--:|--:|--:|--:|
| 1 | 444 | 0.182 | 0.081 |
| 2 | 444 | 0.274 | 0.106 |
| 3 | 444 | 0.341 | 0.162 |
| 4 | 444 | 0.403 | 0.196 |
| 5 | 444 | 0.456 | 0.273 |
| 6 | 444 | 0.505 | 0.275 |
| 7 | 444 | 0.557 | 0.302 |
| 8 | 444 | 0.611 | 0.338 |
| 9 | 444 | 0.666 | 0.435 |
| 10 | 444 | 0.761 | 0.493 |

### DEPOSIT

| decile | n | mean predicted P | observed rate |
|--:|--:|--:|--:|
| 1 | 386 | 0.208 | 0.085 |
| 2 | 386 | 0.286 | 0.106 |
| 3 | 385 | 0.334 | 0.122 |
| 4 | 386 | 0.382 | 0.158 |
| 5 | 385 | 0.428 | 0.166 |
| 6 | 386 | 0.476 | 0.215 |
| 7 | 385 | 0.533 | 0.236 |
| 8 | 386 | 0.593 | 0.326 |
| 9 | 385 | 0.661 | 0.395 |
| 10 | 386 | 0.762 | 0.518 |

### INVESTMENT

| decile | n | mean predicted P | observed rate |
|--:|--:|--:|--:|
| 1 | 523 | 0.142 | 0.055 |
| 2 | 522 | 0.218 | 0.079 |
| 3 | 522 | 0.277 | 0.128 |
| 4 | 522 | 0.336 | 0.153 |
| 5 | 522 | 0.395 | 0.201 |
| 6 | 522 | 0.457 | 0.195 |
| 7 | 522 | 0.526 | 0.280 |
| 8 | 522 | 0.608 | 0.352 |
| 9 | 522 | 0.701 | 0.473 |
| 10 | 523 | 0.856 | 0.677 |
