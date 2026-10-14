
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE, "data", "world_cup_2026_team_prematch_features.csv"
)

df = pd.read_csv(DATA_FILE)

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

final_predictors = [
    "avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "team_fifa_points",
    "opponent_fifa_points",
    "win_rate_before",
    "opponent_scoring_match_rate_before",
    "avg_goal_difference_before",
    "opponent_clean_sheet_rate_before"
]

predictor_sets = {
    "Original": original,
    "Final": final_predictors
}

required = ["match_id", "goals_scored"]
for predictors in predictor_sets.values():
    required.extend(predictors)

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Missing values detected.")

if len(df) != 208 or df["match_id"].nunique() != 104:
    raise ValueError("Unexpected dataset dimensions.")

if not df.groupby("match_id").size().eq(2).all():
    raise ValueError("Every match must have two observations.")

results = []

for test_size in [0.20, 0.30]:

    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=test_size,
        random_state=42
    )

    for split_number, (train_idx, test_idx) in enumerate(
        splitter.split(df, groups=df["match_id"]), start=1
    ):

        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        train_std = train["goals_scored"].std(ddof=0)

        for name, predictors in predictor_sets.items():

            model = LinearRegression()

            model.fit(
                train[predictors],
                train["goals_scored"]
            )

            predictions = model.predict(test[predictors])

            r2 = r2_score(
                test["goals_scored"],
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    test["goals_scored"],
                    predictions
                )
            )

            nrmse = rmse / train_std if train_std > 0 else np.nan

            results.append({
                "Test Size": f"{int(test_size * 100)}%",
                "Split": split_number,
                "Predictor Set": name,
                "R2": r2,
                "RMSE": rmse,
                "NRMSE": nrmse
            })

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby(["Test Size", "Predictor Set"])
    .agg(
        Mean_R2=("R2", "mean"),
        SD_R2=("R2", "std"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_NRMSE=("NRMSE", "mean")
    )
    .round(4)
)

print("\nREGRESSION 2.2 - PREDICTOR SET COMPARISON")
print("=" * 70)
print(summary.to_string())

print("\nTotal evaluations:", len(results_df))
