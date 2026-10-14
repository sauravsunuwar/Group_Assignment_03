# HIT140 Regression 2.1 — Scientific review and change log

**Audit date:** 13 October 2026  
**Evidence:** zipped existing aligned project, corrected and original data, reference ranking/pre-tournament sources, 200-repeat saved outputs, old Markdown and Word reports, revised executed Python outputs.

**HD readiness:** not certifiable; official HIT140 2026 Assessment 3 rubric and full feature provenance missing.

## Critical findings

1. Original `make_pipeline(StandardScaler(), RidgeCV(cv=None))` fits scaling on the full outer train before leave-one-out predictions used inside RidgeCV. It does not leak outer test data but *does* share preprocessing statistics across inner CV folds. Replace with `GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [...]}, cv=KFold(...))`. All inner folds now fit scaling separately.
2. Earlier principal 200-repeat CSV contained only dummy, OLS and Ridge. Original optional outputs show Lasso, Random Forest and Gradient Boosting on seed 42 but no equivalent 200 repeated competitor evaluation. Now all six models share each holdout and each of the 200 repeated seeds, with training-only five-fold hyperparameter searches.
3. Earlier model performance omitted MAE. Revised model-details and repeated splits contain R², RMSE in goals, NRMSE (fraction), MAE in goals, including both train and test where relevant.
4. EDA feature-target correlations are now calculated using the seed-42 80/20 *training* set only. Full-sample diagnostic exploration of outcome relationships should not be used for tuning/selection.
5. Independent re-execution of `regression_2_1_fix_baseline_leakage.py` reconstructed the corrected CSV exactly: nine changed fixtures, 52 adjusted feature cells, no changes to targets or unrelated rows. Six feature columns remain unverified for *all* match dates without the upstream feature-building pipeline.
6. The existing Word/Markdown report describes only three primary models, an earlier VIF summary (full-data maximum 4.31), and an optional competitor comparison. It must be replaced or reconciled with the revised scored six-model results and training-only VIF (~6.32).

## Ranking/decision note

Lasso has the highest mean held-out R² across both 200-split ratios, but zeros some approved coefficients. If the eight-predictor policy requires every coefficient to remain active, keep Ridge as the conservative primary final model and present Lasso as an explicitly conditional competitor. Neither the small holdout nor correlated repeated resamples proves generalisation to new tournaments.

## Official HD rubric status (provisional themes, NOT lecturer rubric wording)

| Domain | Evidence/status | Remaining action |
|---|---|---|
| Data validity | Partially satisfied: uniqueness, finite values, target, FIFA team order, corrected nine fixtures independently reproduced | Upstream feature construction and provenance |
| Methods and leakage | Satisfied for revised in-dataset protocol: nested preprocessing and tuning | Confirm teacher instructions and full timeline |
| Baseline and competitor evaluation | Satisfied within locked 8-feature matrix: six models, two ratios, 200 repeats | Discuss constrained tuning grid |
| Metrics and robustness | Satisfied: train/test R², NRMSE, RMSE, MAE and repeated mean/SD | Confirm team-wide NRMSE denominator conventions |
| EDA and diagnostics | Strengthened: training-only correlations, VIF, residual, Q–Q, BP, Cook’s distance | Independent reading and justification in final group narrative |
| Communication and reproducibility | Partially satisfied: code, CSVs, figures, updated Word report, metadata | Ensure GitHub uses reviewed script and actual rubric & references |
| Formal HD criteria | Unverified: exact official rubric absent | Upload lecturer Yakub’s Assessment 3 marking rubric |

## Exact PowerShell execution

```powershell
python -m pip install pandas numpy scipy scikit-learn statsmodels matplotlib joblib
python regression_2_1_modeling_reviewed.py --data regression_2_1_dataset_corrected.csv --output results_reviewed --repeats 200 --jobs 2
```

The main CSV preserves the six shared columns `model, split, R2_train, R2_test, NRMSE_train, NRMSE_test`; RMSE/MAE and parameter choices are in `regression_2_1_model_details.csv` and `regression_2_1_repeated_summary.csv`.
