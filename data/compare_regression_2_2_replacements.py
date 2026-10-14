import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(
        BASE_DIR,
        "world_cup_2026_team_prematch_features.csv"
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "world_cup_2026_team_prematch_features.csv"
    )
]

FEATURE_FILE = None

for path in possible_paths:
    if os.path.exists(path):
        FEATURE_FILE = path
        break

if FEATURE_FILE is None:
    raise FileNotFoundError(
        "Could not find world_cup_2026_team_prematch_features.csv"
    )


df = pd.read_csv(FEATURE_FILE)

print("=" * 75)
print("REGRESSION 2.2 - CANDIDATE REPLACEMENT TEST")
print("=" * 75)
print("Rows:", len(df))


# ============================================================
# TARGET
# ============================================================

target = "goals_scored"


# ============================================================
# SEVEN VARIABLES THAT STAY FIXED
# ============================================================

fixed_predictors = [
    "avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "team_fifa_points",
    "opponent_fifa_points",
    "win_rate_before",
    "opponent_scoring_match_rate_before",
    "opponent_clean_sheet_rate_before"
]


# ============================================================
# VARIABLE BEING COMPARED
# ============================================================

candidates = [
    "matches_played_before",
    "opponent_avg_goal_difference_before",
    "opponent_points_per_match_before",
    "avg_goal_difference_before"
]


# ============================================================
# MODEL TEST FUNCTION
# ============================================================

def test_model(candidate, test_size):

    predictors = fixed_predictors + [candidate]

    model_df = df[
        ["match_id", target] + predictors
    ].dropna().copy()

    X = model_df[predictors]
    y = model_df[target]

    # Split by MATCH ID so both teams from the same match
    # stay together and do not leak across train/test sets.
    match_ids = model_df["match_id"].unique()

    train_matches, test_matches = train_test_split(
        match_ids,
        test_size=test_size,
        random_state=42
    )

    train_mask = model_df["match_id"].isin(train_matches)
    test_mask = model_df["match_id"].isin(test_matches)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # Normalised RMSE using test-set standard deviation.
    # Lower is better.
    nrmse = rmse / y_test.std(ddof=1)

    return r2, rmse, nrmse


# ============================================================
# VIF FUNCTION
# ============================================================

def calculate_max_vif(candidate):

    predictors = fixed_predictors + [candidate]

    X = df[predictors].dropna().copy()

    vif_values = []

    for i in range(len(predictors)):

        vif = variance_inflation_factor(
            X.values,
            i
        )

        vif_values.append(vif)

    return max(vif_values)


# ============================================================
# RUN COMPARISON
# ============================================================

results = []

for candidate in candidates:

    r2_80, rmse_80, nrmse_80 = test_model(
        candidate,
        0.20
    )

    r2_70, rmse_70, nrmse_70 = test_model(
        candidate,
        0.30
    )

    max_vif = calculate_max_vif(candidate)

    correlation = df[
        [candidate, target]
    ].corr().iloc[0, 1]

    results.append({
        "candidate": candidate,
        "target_correlation": correlation,
        "R2_80_20": r2_80,
        "NRMSE_80_20": nrmse_80,
        "R2_70_30": r2_70,
        "NRMSE_70_30": nrmse_70,
        "max_VIF": max_vif
    })


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY
# ============================================================

print("\nCandidate comparison:\n")

print(
    results_df.to_string(
        index=False
    )
)


print("\n" + "=" * 75)

print(
    "Interpretation:"
)

print(
    "- Higher R² is better."
)

print(
    "- Lower NRMSE is better."
)

print(
    "- Lower VIF is better; values below 5 are generally preferable."
)

print(
    "- Do not select a variable from correlation alone."
)

print("=" * 75)