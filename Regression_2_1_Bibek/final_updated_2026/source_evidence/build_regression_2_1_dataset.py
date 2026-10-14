import os
import numpy as np
import pandas as pd


# ============================================================
# HIT140 ASSESSMENT 3 - REGRESSION 2.1
# Corrected dataset builder
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(filename):
    for folder in [BASE_DIR, os.path.join(BASE_DIR, "data")]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Cannot find {filename}")


MASTER_FILE = find_file("world_cup_2026_master_matches.csv")
FEATURE_FILE = find_file(
    "world_cup_2026_team_prematch_features.csv"
)

OUTPUT_FILE = os.path.join(
    os.path.dirname(MASTER_FILE),
    "regression_2_1_dataset.csv"
)


# ============================================================
# 1. LOAD AND VALIDATE SOURCE DATA
# ============================================================

print("=" * 72)
print("HIT140 - REGRESSION 2.1 DATASET BUILDER")
print("=" * 72)

master_df = pd.read_csv(MASTER_FILE)
features_df = pd.read_csv(FEATURE_FILE)

print("Master matches:", len(master_df))
print("Team-match feature rows:", len(features_df))

if len(master_df) != 104:
    raise ValueError("Expected exactly 104 matches.")

if master_df["match_id"].nunique() != 104:
    raise ValueError("Expected 104 unique match IDs.")

if len(features_df) != 208:
    raise ValueError("Expected 208 team-match feature rows.")

if features_df.duplicated(["match_id", "team"]).any():
    raise ValueError("Duplicate team-match feature rows.")


# ============================================================
# 2. SELECT REQUIRED PRE-MATCH FEATURES
# ============================================================

feature_columns = [
    "match_id",
    "team",
    "team_fifa_points",
    "points_per_match_before",
    "avg_goals_scored_before",
    "avg_goals_conceded_before",
    "scoring_match_rate_before",
    "scoring_consistency_before",
    "clean_sheet_rate_before"
]

master_columns = [
    "match_id",
    "match_date",
    "stage",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals"
]

for name, df, columns in [
    ("Master", master_df, master_columns),
    ("Features", features_df, feature_columns)
]:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} dataset missing columns: {missing}"
        )


# ============================================================
# 3. PREPARE HOME AND AWAY FEATURES
# ============================================================

home_features = features_df[feature_columns].rename(
    columns={
        "team": "home_team",
        **{
            c: "home_" + c
            for c in feature_columns
            if c not in ["match_id", "team"]
        }
    }
)

away_features = features_df[feature_columns].rename(
    columns={
        "team": "away_team",
        **{
            c: "away_" + c
            for c in feature_columns
            if c not in ["match_id", "team"]
        }
    }
)


# ============================================================
# 4. MERGE HOME AND AWAY DATA
# ============================================================

match_df = master_df[master_columns].copy()

match_df = match_df.merge(
    home_features,
    on=["match_id", "home_team"],
    how="left",
    validate="one_to_one"
)

match_df = match_df.merge(
    away_features,
    on=["match_id", "away_team"],
    how="left",
    validate="one_to_one"
)

if len(match_df) != 104:
    raise ValueError("Feature merge changed the row count.")

merged_feature_columns = [
    c for c in match_df.columns
    if c.startswith(("home_", "away_"))
    and c not in [
        "home_team",
        "away_team",
        "home_goals",
        "away_goals"
    ]
]

if match_df[merged_feature_columns].isna().any().any():
    raise ValueError(
        "Missing home or away pre-match features."
    )

print("Home/away feature integration: PASSED")


# ============================================================
# 5. TARGET VARIABLE
# ============================================================

match_df["goal_difference"] = (
    match_df["home_goals"]
    - match_df["away_goals"]
)


# ============================================================
# 6. CALCULATE THE SEVEN NUMERIC DIFFERENCES
# ============================================================

feature_mapping = {
    "fifa_points_difference": "team_fifa_points",
    "points_per_match_difference": "points_per_match_before",
    "avg_goals_scored_difference": "avg_goals_scored_before",
    "avg_goals_conceded_difference": "avg_goals_conceded_before",
    "scoring_match_rate_difference": "scoring_match_rate_before",
    "scoring_consistency_difference": "scoring_consistency_before",
    "clean_sheet_rate_difference": "clean_sheet_rate_before"
}

for output_name, source_name in feature_mapping.items():
    match_df[output_name] = (
        match_df["home_" + source_name]
        - match_df["away_" + source_name]
    )


# ============================================================
# 7. HOST-COUNTRY INDICATOR
# ============================================================

HOST_COUNTRIES = {"Canada", "Mexico", "USA"}

match_df["host_country_indicator"] = (
    match_df["home_team"].isin(HOST_COUNTRIES).astype(int)
    - match_df["away_team"].isin(HOST_COUNTRIES).astype(int)
)


# ============================================================
# 8. EXACTLY EIGHT PREDICTORS
# ============================================================

predictors = [
    "fifa_points_difference",
    "points_per_match_difference",
    "avg_goals_scored_difference",
    "avg_goals_conceded_difference",
    "scoring_match_rate_difference",
    "scoring_consistency_difference",
    "clean_sheet_rate_difference",
    "host_country_indicator"
]

assert len(predictors) == 8


# ============================================================
# 9. CREATE FINAL DATASET
# ============================================================

output_columns = (
    master_columns
    + ["goal_difference"]
    + predictors
)

regression_df = match_df[output_columns].copy()


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 72)
print("FINAL VALIDATION")
print("=" * 72)

print("Rows:", len(regression_df))
print("Unique matches:", regression_df["match_id"].nunique())
print("Predictors:", len(predictors))

if len(regression_df) != 104:
    raise ValueError("Incorrect row count.")

if regression_df["match_id"].nunique() != 104:
    raise ValueError("Incorrect unique-match count.")

if regression_df[predictors].isna().any().any():
    raise ValueError("Missing predictor values.")

if regression_df["goal_difference"].isna().any():
    raise ValueError("Missing target values.")

constant_columns = [
    c for c in predictors
    if regression_df[c].nunique() <= 1
]

if constant_columns:
    raise ValueError(
        f"Constant predictors: {constant_columns}"
    )

# Check for exact duplicate or opposite predictors.
for i, first in enumerate(predictors):
    for second in predictors[i + 1:]:
        a = regression_df[first].to_numpy(dtype=float)
        b = regression_df[second].to_numpy(dtype=float)

        if np.allclose(a, b) or np.allclose(a, -b):
            raise ValueError(
                f"Redundant predictors: {first}, {second}"
            )

print("Structural validation: PASSED")
print("Missing-value validation: PASSED")
print("Constant/redundancy checks: PASSED")


# ============================================================
# 11. CORRELATION CHECKS
# ============================================================

correlations = regression_df[predictors].corr()

print("\nPREDICTOR CORRELATION MATRIX")
print(correlations.round(3).to_string())

print("\nHIGH-CORRELATION PAIRS (|r| >= 0.80)")

high_pairs = []

for i, first in enumerate(predictors):
    for second in predictors[i + 1:]:
        r = correlations.loc[first, second]

        if abs(r) >= 0.80:
            high_pairs.append((first, second, r))

if high_pairs:
    for first, second, r in high_pairs:
        print(f"{first} <-> {second}: {r:.3f}")
else:
    print("None")

print("\nCORRELATION WITH GOAL DIFFERENCE")

target_correlations = (
    regression_df[predictors + ["goal_difference"]]
    .corr()["goal_difference"]
    .drop("goal_difference")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

print(target_correlations.round(3).to_string())


# ============================================================
# 12. SAVE
# ============================================================

regression_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 72)
print("REGRESSION 2.1 DATASET SAVED")
print("=" * 72)

print("Saved to:", OUTPUT_FILE)
print("Rows:", len(regression_df))
print("Unique matches:", regression_df["match_id"].nunique())
print("Predictors:", len(predictors))
print("Target: goal_difference")
print("\nVALIDATION: PASSED")