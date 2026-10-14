import os
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MASTER_FILE = os.path.join(
    DATA_DIR,
    "world_cup_2026_master_matches.csv"
)

TEAM_FEATURE_FILE = os.path.join(
    DATA_DIR,
    "world_cup_2026_team_prematch_features.csv"
)

BASELINE_FILE = os.path.join(
    DATA_DIR,
    "pre_tournament_baselines.csv"
)

RANKING_FILE = os.path.join(
    DATA_DIR,
    "fifa_rankings_11_june_2026.csv"
)


# ============================================================
# 2. LOAD MASTER DATABASE
# ============================================================

print("=" * 70)
print("HIT140 ASSESSMENT 3 - PRE-MATCH FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading master database...")

df = pd.read_csv(MASTER_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# 3. VALIDATE MASTER DATABASE
# ============================================================

required_columns = [
    "match_id",
    "match_date",
    "stage",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_goals",
    "away_goals",
    "goal_difference"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if len(df) != 104:
    raise ValueError(
        f"Expected 104 matches, but found {len(df)}."
    )

if df["match_id"].duplicated().sum() > 0:
    raise ValueError(
        "Duplicate match IDs found."
    )

print("Master database validation: PASSED")


# ============================================================
# 4. PREPARE DATE AND SORT CHRONOLOGICALLY
# ============================================================

df["match_date"] = pd.to_datetime(
    df["match_date"],
    errors="coerce",
    utc=True
)

if df["match_date"].isna().any():
    raise ValueError(
        "Some match dates could not be converted."
    )

df = df.sort_values(
    ["match_date", "match_id"]
).reset_index(drop=True)

print("Chronological sorting: COMPLETE")


# ============================================================
# 5. CONVERT 104 MATCHES INTO 208 TEAM-MATCH OBSERVATIONS
# ============================================================

team_rows = []

for _, row in df.iterrows():

    # --------------------------------------------------------
    # HOME TEAM
    # --------------------------------------------------------

    if row["home_goals"] > row["away_goals"]:
        home_points = 3
        home_win = 1
    elif row["home_goals"] == row["away_goals"]:
        home_points = 1
        home_win = 0
    else:
        home_points = 0
        home_win = 0

    home_scored = int(row["home_goals"] > 0)
    home_clean_sheet = int(row["away_goals"] == 0)

    team_rows.append({
        "match_id": row["match_id"],
        "match_date": row["match_date"],
        "stage": row["stage"],

        "team_id": row["home_team_id"],
        "team": row["home_team"],

        "opponent_id": row["away_team_id"],
        "opponent": row["away_team"],

        "home_away": "Home",

        "goals_scored": row["home_goals"],
        "goals_conceded": row["away_goals"],

        "match_goal_difference":
            row["home_goals"] - row["away_goals"],

        "points_earned": home_points,
        "win": home_win,

        # New binary outcomes used only to construct
        # leakage-free historical predictors.
        "scored_in_match": home_scored,
        "clean_sheet": home_clean_sheet
    })

    # --------------------------------------------------------
    # AWAY TEAM
    # --------------------------------------------------------

    if row["away_goals"] > row["home_goals"]:
        away_points = 3
        away_win = 1
    elif row["away_goals"] == row["home_goals"]:
        away_points = 1
        away_win = 0
    else:
        away_points = 0
        away_win = 0

    away_scored = int(row["away_goals"] > 0)
    away_clean_sheet = int(row["home_goals"] == 0)

    team_rows.append({
        "match_id": row["match_id"],
        "match_date": row["match_date"],
        "stage": row["stage"],

        "team_id": row["away_team_id"],
        "team": row["away_team"],

        "opponent_id": row["home_team_id"],
        "opponent": row["home_team"],

        "home_away": "Away",

        "goals_scored": row["away_goals"],
        "goals_conceded": row["home_goals"],

        "match_goal_difference":
            row["away_goals"] - row["home_goals"],

        "points_earned": away_points,
        "win": away_win,

        "scored_in_match": away_scored,
        "clean_sheet": away_clean_sheet
    })


team_df = pd.DataFrame(team_rows)

team_df = team_df.sort_values(
    ["match_date", "match_id", "team"]
).reset_index(drop=True)

print(
    f"Team-match observations created: {len(team_df)}"
)

if len(team_df) != 208:
    raise ValueError(
        f"Expected 208 team-match observations, "
        f"but found {len(team_df)}."
    )


# ============================================================
# 6. CALCULATE PRE-MATCH TOURNAMENT FEATURES
# ============================================================
#
# CRITICAL LEAKAGE PROTECTION:
#
# shift(1) removes the CURRENT match before calculating
# historical statistics.
#
# Therefore every predictor below uses only matches that
# occurred before the target match.
# ============================================================

team_df["matches_played_before"] = (
    team_df.groupby("team_id")
    .cumcount()
)


# ------------------------------------------------------------
# Prior average goals scored
# ------------------------------------------------------------

team_df["avg_goals_scored_before"] = (
    team_df.groupby("team_id")["goals_scored"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ------------------------------------------------------------
# Prior average goals conceded
# ------------------------------------------------------------

team_df["avg_goals_conceded_before"] = (
    team_df.groupby("team_id")["goals_conceded"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ------------------------------------------------------------
# Prior win rate
# ------------------------------------------------------------

team_df["win_rate_before"] = (
    team_df.groupby("team_id")["win"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ------------------------------------------------------------
# Prior points per match
# ------------------------------------------------------------

team_df["points_per_match_before"] = (
    team_df.groupby("team_id")["points_earned"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ------------------------------------------------------------
# Prior average goal difference
# ------------------------------------------------------------

team_df["avg_goal_difference_before"] = (
    team_df.groupby("team_id")["match_goal_difference"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ============================================================
# NEW FEATURE 1:
# PRIOR SCORING CONSISTENCY / VOLATILITY
# ============================================================
#
# Standard deviation of goals scored in PRIOR matches.
#
# A lower value means the team's scoring output has been
# relatively consistent.
#
# ddof=0 allows a value of 0 when only one prior tournament
# match is available.
# ============================================================

team_df["scoring_consistency_before"] = (
    team_df.groupby("team_id")["goals_scored"]
    .transform(
        lambda x: (
            x.shift(1)
            .expanding()
            .std(ddof=0)
        )
    )
)


# ============================================================
# NEW FEATURE 2:
# PRIOR SCORING-MATCH RATE
# ============================================================
#
# Proportion of PRIOR matches in which the team scored
# at least one goal.
#
# Example:
# scored in 4 of previous 5 matches -> 0.80
# ============================================================

team_df["scoring_match_rate_before"] = (
    team_df.groupby("team_id")["scored_in_match"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ============================================================
# NEW FEATURE 3:
# PRIOR CLEAN-SHEET RATE
# ============================================================
#
# Proportion of PRIOR matches in which the team conceded
# zero goals.
# ============================================================

team_df["clean_sheet_rate_before"] = (
    team_df.groupby("team_id")["clean_sheet"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ============================================================
# 7. IDENTIFY FIRST TOURNAMENT MATCHES
# ============================================================

team_df["first_tournament_match"] = (
    team_df["matches_played_before"] == 0
)

first_match_count = (
    team_df["first_tournament_match"].sum()
)

print(
    f"First-match observations: {first_match_count}"
)

if first_match_count != 48:
    raise ValueError(
        f"Expected 48 first-match observations, "
        f"but found {first_match_count}."
    )


# ============================================================
# 8. LOAD PRE-TOURNAMENT BASELINES
# ============================================================

print("\nLoading pre-tournament baselines...")

baseline_df = pd.read_csv(BASELINE_FILE)

print(
    "Baseline teams loaded:",
    len(baseline_df)
)


# ============================================================
# 9. VALIDATE BASELINE FILE
# ============================================================

required_baseline_columns = [
    "team",
    "baseline_avg_goals_scored",
    "baseline_avg_goals_conceded",
    "baseline_win_rate",
    "baseline_points_per_match",
    "baseline_avg_goal_difference",

    # NEW baseline variables
    "baseline_scoring_consistency",
    "baseline_scoring_match_rate",
    "baseline_clean_sheet_rate"
]

missing_baseline_columns = [
    column
    for column in required_baseline_columns
    if column not in baseline_df.columns
]

if missing_baseline_columns:
    raise ValueError(
        "\nYour baseline file has not yet been updated.\n"
        "Missing baseline columns:\n"
        f"{missing_baseline_columns}\n\n"
        "This is expected until "
        "build_pretournament_baseline.py is updated."
    )

if baseline_df["team"].nunique() != 48:
    raise ValueError(
        "Expected baseline data for exactly 48 teams."
    )

if baseline_df["team"].duplicated().any():
    raise ValueError(
        "Duplicate teams detected in baseline file."
    )


# ============================================================
# 10. CHECK TEAM-NAME MATCHING
# ============================================================

world_cup_team_names = set(
    team_df["team"].unique()
)

baseline_team_names = set(
    baseline_df["team"].unique()
)

missing_baseline_teams = sorted(
    world_cup_team_names
    - baseline_team_names
)

print(
    "World Cup teams missing baseline data:",
    len(missing_baseline_teams)
)

if missing_baseline_teams:

    for team in missing_baseline_teams:
        print(" -", team)

    raise ValueError(
        "Some World Cup teams do not have "
        "pre-tournament baseline data."
    )


# ============================================================
# 11. MERGE BASELINE STATISTICS
# ============================================================

team_df = team_df.merge(
    baseline_df[
        required_baseline_columns
    ],
    on="team",
    how="left"
)


# ============================================================
# 12. FILL ONLY FIRST-MATCH FEATURES
# ============================================================

first_match_mask = (
    team_df["first_tournament_match"]
)


baseline_mapping = {

    "avg_goals_scored_before":
        "baseline_avg_goals_scored",

    "avg_goals_conceded_before":
        "baseline_avg_goals_conceded",

    "win_rate_before":
        "baseline_win_rate",

    "points_per_match_before":
        "baseline_points_per_match",

    "avg_goal_difference_before":
        "baseline_avg_goal_difference",

    "scoring_consistency_before":
        "baseline_scoring_consistency",

    "scoring_match_rate_before":
        "baseline_scoring_match_rate",

    "clean_sheet_rate_before":
        "baseline_clean_sheet_rate"
}


for feature, baseline_feature in baseline_mapping.items():

    team_df.loc[
        first_match_mask,
        feature
    ] = team_df.loc[
        first_match_mask,
        baseline_feature
    ]


# ============================================================
# 13. VALIDATE BASELINE INTEGRATION
# ============================================================

rolling_features = [
    "avg_goals_scored_before",
    "avg_goals_conceded_before",
    "win_rate_before",
    "points_per_match_before",
    "avg_goal_difference_before",
    "scoring_consistency_before",
    "scoring_match_rate_before",
    "clean_sheet_rate_before"
]

remaining_missing = (
    team_df[rolling_features]
    .isna()
    .sum()
)

print(
    "\nMissing pre-match features after "
    "baseline integration:"
)

print(remaining_missing)

if remaining_missing.sum() != 0:
    raise ValueError(
        "Missing pre-match features remain after "
        "baseline integration."
    )

print(
    "\nFirst-match baseline integration: PASSED"
)


# ============================================================
# 14. LOAD FIFA PRE-TOURNAMENT RANKINGS
# ============================================================

print("\nLoading FIFA pre-tournament rankings...")

ranking_df = pd.read_csv(RANKING_FILE)

required_ranking_columns = [
    "team",
    "fifa_rank",
    "fifa_points"
]

missing_ranking_columns = [
    column
    for column in required_ranking_columns
    if column not in ranking_df.columns
]

if missing_ranking_columns:
    raise ValueError(
        f"FIFA ranking file missing columns: "
        f"{missing_ranking_columns}"
    )

print(
    "FIFA ranking teams loaded:",
    ranking_df["team"].nunique()
)

world_cup_teams = set(
    team_df["team"].unique()
)

ranking_teams = set(
    ranking_df["team"].unique()
)

missing_from_rankings = sorted(
    world_cup_teams - ranking_teams
)

extra_ranking_teams = sorted(
    ranking_teams - world_cup_teams
)

print(
    "\nWorld Cup teams missing from "
    "FIFA ranking file:"
)

if missing_from_rankings:
    for team in missing_from_rankings:
        print(f" - {team}")
else:
    print("None")

print(
    "\nRanking teams not found in "
    "World Cup database:"
)

if extra_ranking_teams:
    for team in extra_ranking_teams:
        print(f" - {team}")
else:
    print("None")

if missing_from_rankings:
    raise ValueError(
        "Some World Cup teams do not have FIFA ranking data. "
        "Fix the team names before continuing."
    )


# ============================================================
# 15. MERGE TEAM FIFA RANKING
# ============================================================

team_df = team_df.merge(
    ranking_df[
        [
            "team",
            "fifa_rank",
            "fifa_points"
        ]
    ],
    on="team",
    how="left"
)

team_df = team_df.rename(
    columns={
        "fifa_rank": "team_fifa_rank",
        "fifa_points": "team_fifa_points"
    }
)


# ============================================================
# 16. MERGE OPPONENT FIFA RANKING
# ============================================================

opponent_ranking_df = ranking_df.rename(
    columns={
        "team": "opponent",
        "fifa_rank": "opponent_fifa_rank",
        "fifa_points": "opponent_fifa_points"
    }
)

team_df = team_df.merge(
    opponent_ranking_df[
        [
            "opponent",
            "opponent_fifa_rank",
            "opponent_fifa_points"
        ]
    ],
    on="opponent",
    how="left"
)


# ============================================================
# 17. FIFA RANKING VALIDATION
# ============================================================

print("\nFIFA ranking merge validation:")

print(
    "Missing team FIFA points:",
    team_df["team_fifa_points"].isna().sum()
)

print(
    "Missing opponent FIFA points:",
    team_df["opponent_fifa_points"].isna().sum()
)

print(
    "Missing team FIFA ranks:",
    team_df["team_fifa_rank"].isna().sum()
)

print(
    "Missing opponent FIFA ranks:",
    team_df["opponent_fifa_rank"].isna().sum()
)

ranking_missing = team_df[
    [
        "team_fifa_points",
        "opponent_fifa_points",
        "team_fifa_rank",
        "opponent_fifa_rank"
    ]
].isna().sum().sum()

if ranking_missing != 0:
    raise ValueError(
        "Missing FIFA ranking values remain."
    )

print("\nFIFA ranking integration: COMPLETE")


# ============================================================
# 18. CREATE OPPONENT PRE-MATCH FEATURES
# ============================================================

print(
    "\nCreating opponent pre-match features..."
)

opponent_features = team_df[
    [
        "match_id",
        "team_id",
        "team",

        "avg_goals_scored_before",
        "avg_goals_conceded_before",
        "win_rate_before",
        "points_per_match_before",
        "avg_goal_difference_before",

        "scoring_consistency_before",
        "scoring_match_rate_before",
        "clean_sheet_rate_before",

        "matches_played_before"
    ]
].copy()


opponent_features = opponent_features.rename(
    columns={

        "team_id":
            "opponent_id",

        "team":
            "opponent_check",

        "avg_goals_scored_before":
            "opponent_avg_goals_scored_before",

        "avg_goals_conceded_before":
            "opponent_avg_goals_conceded_before",

        "win_rate_before":
            "opponent_win_rate_before",

        "points_per_match_before":
            "opponent_points_per_match_before",

        "avg_goal_difference_before":
            "opponent_avg_goal_difference_before",

        "scoring_consistency_before":
            "opponent_scoring_consistency_before",

        "scoring_match_rate_before":
            "opponent_scoring_match_rate_before",

        "clean_sheet_rate_before":
            "opponent_clean_sheet_rate_before",

        "matches_played_before":
            "opponent_matches_played_before"
    }
)


team_df = team_df.merge(
    opponent_features,
    on=[
        "match_id",
        "opponent_id"
    ],
    how="left"
)


# ============================================================
# 19. VALIDATE OPPONENT FEATURES
# ============================================================

print("\nOpponent feature validation:")

print(
    "Rows after merge:",
    len(team_df)
)

duplicate_count = team_df.duplicated(
    subset=[
        "match_id",
        "team_id"
    ]
).sum()

print(
    "Duplicate team-match rows:",
    duplicate_count
)

if len(team_df) != 208:
    raise ValueError(
        f"Expected 208 rows after opponent merge, "
        f"but found {len(team_df)}."
    )

if duplicate_count != 0:
    raise ValueError(
        "Duplicate team-match rows found "
        "after opponent merge."
    )

opponent_mismatch = (
    team_df["opponent"]
    != team_df["opponent_check"]
).sum()

print(
    "Opponent name mismatches:",
    opponent_mismatch
)

if opponent_mismatch != 0:
    raise ValueError(
        "Opponent matching error detected."
    )

team_df = team_df.drop(
    columns=["opponent_check"]
)


opponent_feature_columns = [
    "opponent_avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "opponent_win_rate_before",
    "opponent_points_per_match_before",
    "opponent_avg_goal_difference_before",
    "opponent_scoring_consistency_before",
    "opponent_scoring_match_rate_before",
    "opponent_clean_sheet_rate_before"
]

opponent_missing = (
    team_df[opponent_feature_columns]
    .isna()
    .sum()
)

print(
    "\nMissing opponent pre-match features:"
)

print(opponent_missing)

if opponent_missing.sum() != 0:
    raise ValueError(
        "Missing opponent pre-match "
        "features remain."
    )

print(
    "\nOpponent pre-match features: COMPLETE"
)


# ============================================================
# 20. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    "Rows:",
    len(team_df)
)

print(
    "Unique matches:",
    team_df["match_id"].nunique()
)

print(
    "Unique teams:",
    team_df["team_id"].nunique()
)

print(
    "Duplicate team-match rows:",
    team_df.duplicated(
        subset=[
            "match_id",
            "team_id"
        ]
    ).sum()
)


if len(team_df) != 208:
    raise ValueError(
        "Final dataset must contain 208 rows."
    )

if team_df["match_id"].nunique() != 104:
    raise ValueError(
        "Final dataset must contain 104 matches."
    )

if team_df["team_id"].nunique() != 48:
    raise ValueError(
        "Final dataset must contain 48 teams."
    )


# ============================================================
# 21. DISPLAY EXAMPLE
# ============================================================

display_columns = [
    "match_id",
    "match_date",
    "team",
    "opponent",
    "goals_scored",

    "avg_goals_scored_before",
    "avg_goals_conceded_before",

    "scoring_consistency_before",
    "scoring_match_rate_before",
    "clean_sheet_rate_before",

    "team_fifa_points",

    "opponent_avg_goals_conceded_before",
    "opponent_clean_sheet_rate_before",
    "opponent_fifa_points"
]

print("\nExample observations:")

print(
    team_df[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 22. SAVE UPDATED TEAM PRE-MATCH DATABASE
# ============================================================

team_df.to_csv(
    TEAM_FEATURE_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print(
    f"\nSaved to:\n{TEAM_FEATURE_FILE}"
)

print(
    "\nPre-match feature engineering "
    "completed successfully."
)