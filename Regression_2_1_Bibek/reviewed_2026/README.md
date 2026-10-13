Regression 2.1 – Goal Difference Modelling
HIT140 Foundations of Data Science – Assessment 3

Contributor: Bibek

Status: Reviewed modelling update, pending final team approval.

Project Overview
This project predicts FIFA World Cup 2026 match goal difference using 104 match observations and eight approved pre-match predictors.

The target variable is calculated as:

goal_difference = home_goals − away_goals

Models Evaluated
Mean Baseline, Multiple Linear Regression, Ridge Regression, Lasso Regression, Random Forest and Gradient Boosting.

Evaluation Methodology
The analysis uses 70/30 and 80/20 train/test splits with random seed 42.

Each split ratio is evaluated using 200 repeated random splits.

Hyperparameter tuning uses five-fold cross-validation within the training data. Scaling for Ridge and Lasso is performed within cross-validation pipelines to prevent preprocessing leakage.

Performance is evaluated using R², RMSE, MAE and NRMSE.

NRMSE = RMSE / (maximum − minimum observed target value in the evaluated partition).

Requirements
Python with pandas, numpy, scikit-learn, scipy, statsmodels and matplotlib.

How to Run
Open PowerShell inside the reviewed_2026 folder.

Install the required libraries if necessary:

python -m pip install pandas numpy scikit-learn scipy statsmodels matplotlib

Run the reviewed modelling script:

python regression_2_1_modeling_reviewed.py --data regression_2_1_dataset_corrected.csv --output results_reviewed

The analysis may take several minutes to complete.

Generated Outputs
The results_reviewed folder contains model comparison results, 200 repeated-split evaluations, predictions, hyperparameter tuning records, exploratory data analysis, statistical diagnostics and visualisations.

The principal result files are regression_2_1_results.csv and regression_2_1_repeated_summary.csv.

Model Recommendation
Lasso achieved the strongest average predictive performance across the repeated splits.

However, Lasso can reduce some predictor coefficients to zero. If all eight predictors must remain active, Ridge Regression is recommended as the primary regularised model.

Final model selection remains subject to team and lecturer confirmation.

Notes
The reviewed analysis uses the approved corrected dataset. Full historical feature provenance and compliance with the official marking rubric require final verification.

The original Regression 2.1 files are retained separately for comparison and reproducibility.
