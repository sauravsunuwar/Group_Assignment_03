import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(
        BASE,
        "data",
        "world_cup_2026_team_prematch_features.csv"
    )
)

original = [
    "avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "scoring_match_rate_before",
    "opponent_clean_sheet_rate_before",
    "team_fifa_points",
    "opponent_fifa_points",
    "scoring_consistency_before",
    "opponent_points_per_match_before"
]

alternative = [
    "avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "team_fifa_points",
    "opponent_fifa_points",
    "win_rate_before",
    "opponent_scoring_match_rate_before",
    "matches_played_before",
    "opponent_clean_sheet_rate_before"
]

cv = GroupKFold(n_splits=5)

scoring = {
    "r2": "r2",
    "rmse": make_scorer(
        lambda actual, predicted: np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        greater_is_better=False
    )
}

for name, predictors in [
    ("Original", original),
    ("Alternative", alternative)
]:
    results = cross_validate(
        LinearRegression(),
        df[predictors],
        df["goals_scored"],
        groups=df["match_id"],
        cv=cv,
        scoring=scoring
    )

    r2 = results["test_r2"]
    rmse = -results["test_rmse"]

    print(f"\n{name.upper()}")
    print("Fold R²:", np.round(r2, 4))
    print("Mean R²:", round(r2.mean(), 4))
    print("Fold RMSE:", np.round(rmse, 4))
    print("Mean RMSE:", round(rmse.mean(), 4))