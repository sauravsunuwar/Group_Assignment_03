import os
import pandas as pd


# ============================================================
# HIT140 ASSESSMENT 3
# PRE-TOURNAMENT BASELINE BUILDER
#
# Purpose:
# Create pre-tournament performance baselines for each of the
# 48 FIFA World Cup 2026 teams.
#
# IMPORTANT:
# Only information available BEFORE each team's first
# World Cup 2026 match may be used.
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "pre_tournament_matches.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "pre_tournament_baselines.csv"
)


# ============================================================
# 2. LOAD PRE-TOURNAMENT MATCHES
# ============================================================

print("=" * 70)
print("HIT140 - PRE-TOURNAMENT BASELINE BUILDER")
print("=" * 70)

print("\nLoading pre-tournament match data...")

df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))
print("Columns loaded:", len(df.columns))


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "match_date",
    "team",
    "opponent",
    "goals_scored",
    "goals_conceded"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns: PASSED")


# ============================================================
# 4. PREPARE MATCH DATES
# ============================================================

df["match_date"] = pd.to_datetime(
    df["match_date"],
    errors="coerce"
)

if df["match_date"].isna().any():
    raise ValueError(
        "Some match dates could not be converted."
    )

print("Match-date conversion: PASSED")


# ============================================================
# 5. DATE VALIDATION NOTE
# ============================================================
#
# build_pretournament_match_data.py is responsible for
# selecting matches that occurred BEFORE each team's first
# World Cup 2026 match.
#
# We therefore do not apply one global tournament cutoff here.
# ============================================================

print(
    "Pre-tournament match dates were previously filtered "
    "against each team's first World Cup match."
)


# ============================================================
# 6. CALCULATE MATCH-LEVEL VARIABLES
# ============================================================

# ------------------------------------------------------------
# Win indicator
# ------------------------------------------------------------

df["win"] = (
    df["goals_scored"]
    > df["goals_conceded"]
).astype(int)


# ------------------------------------------------------------
# Points earned
# ------------------------------------------------------------

def calculate_points(row):

    if row["goals_scored"] > row["goals_conceded"]:
        return 3

    elif row["goals_scored"] == row["goals_conceded"]:
        return 1

    else:
        return 0


df["points"] = df.apply(
    calculate_points,
    axis=1
)


# ------------------------------------------------------------
# Goal difference
# ------------------------------------------------------------

df["goal_difference"] = (
    df["goals_scored"]
    - df["goals_conceded"]
)


# ============================================================
# NEW MATCH-LEVEL VARIABLES
# ============================================================

# ------------------------------------------------------------
# Scored in match
#
# 1 = team scored at least one goal
# 0 = team scored zero goals
# ------------------------------------------------------------

df["scored_in_match"] = (
    df["goals_scored"] > 0
).astype(int)


# ------------------------------------------------------------
# Clean sheet
#
# 1 = team conceded zero goals
# 0 = team conceded at least one goal
# ------------------------------------------------------------

df["clean_sheet"] = (
    df["goals_conceded"] == 0
).astype(int)


# ============================================================
# 7. CALCULATE TEAM BASELINES
# ============================================================
#
# All baseline statistics are calculated from each team's
# selected pre-tournament matches.
# ============================================================

baseline_df = (
    df.groupby("team")
    .agg(

        # Number of historical matches
        baseline_matches=(
            "team",
            "size"
        ),

        # ---------------------------------------------
        # ORIGINAL FEATURES
        # ---------------------------------------------

        baseline_avg_goals_scored=(
            "goals_scored",
            "mean"
        ),

        baseline_avg_goals_conceded=(
            "goals_conceded",
            "mean"
        ),

        baseline_win_rate=(
            "win",
            "mean"
        ),

        baseline_points_per_match=(
            "points",
            "mean"
        ),

        baseline_avg_goal_difference=(
            "goal_difference",
            "mean"
        ),

        # ---------------------------------------------
        # NEW FEATURE 1:
        # SCORING CONSISTENCY / VOLATILITY
        # ---------------------------------------------
        #
        # Standard deviation of goals scored across
        # the team's pre-tournament matches.
        #
        # Lower value = more consistent scoring.
        # Higher value = more variable scoring.
        #
        # pandas groupby std normally uses ddof=1.
        # We calculate this separately below with
        # ddof=0 so that it matches our tournament
        # rolling-feature definition.
        # ---------------------------------------------

        # ---------------------------------------------
        # NEW FEATURE 2:
        # SCORING-MATCH RATE
        # ---------------------------------------------

        baseline_scoring_match_rate=(
            "scored_in_match",
            "mean"
        ),

        # ---------------------------------------------
        # NEW FEATURE 3:
        # CLEAN-SHEET RATE
        # ---------------------------------------------

        baseline_clean_sheet_rate=(
            "clean_sheet",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 8. CALCULATE SCORING CONSISTENCY
# ============================================================
#
# Population standard deviation (ddof=0) is used here.
#
# This is consistent with the definition used later in
# build_prematch_features.py.
# ============================================================

scoring_consistency = (
    df.groupby("team")["goals_scored"]
    .std(ddof=0)
    .reset_index(
        name="baseline_scoring_consistency"
    )
)

baseline_df = baseline_df.merge(
    scoring_consistency,
    on="team",
    how="left"
)


# ============================================================
# 9. VALIDATE EXPECTED DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE VALIDATION")
print("=" * 70)

print(
    "\nTeams with baseline data:",
    len(baseline_df)
)

print(
    "Total team-match observations:",
    len(df)
)


# Exactly 48 World Cup teams should be represented
if len(baseline_df) != 48:
    raise ValueError(
        f"Expected 48 teams, "
        f"but found {len(baseline_df)}."
    )


# Team names must be unique
if baseline_df["team"].duplicated().any():
    raise ValueError(
        "Duplicate teams found in baseline dataset."
    )


# ============================================================
# 10. CHECK NUMBER OF MATCHES PER TEAM
# ============================================================

print(
    "\nBaseline matches per team:"
)

print(
    baseline_df["baseline_matches"]
    .describe()
)


teams_not_five = baseline_df[
    baseline_df["baseline_matches"] != 5
]

print(
    "\nTeams without exactly 5 baseline matches:",
    len(teams_not_five)
)

if len(teams_not_five) > 0:

    print(
        teams_not_five[
            [
                "team",
                "baseline_matches"
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        "Every team should have exactly "
        "5 pre-tournament matches."
    )


# ============================================================
# 11. CHECK MISSING VALUES
# ============================================================

baseline_columns = [
    "baseline_avg_goals_scored",
    "baseline_avg_goals_conceded",
    "baseline_win_rate",
    "baseline_points_per_match",
    "baseline_avg_goal_difference",

    "baseline_scoring_consistency",
    "baseline_scoring_match_rate",
    "baseline_clean_sheet_rate"
]

print(
    "\nMissing baseline values:"
)

missing_values = (
    baseline_df[
        baseline_columns
    ]
    .isna()
    .sum()
)

print(missing_values)

if missing_values.sum() != 0:
    raise ValueError(
        "Missing baseline values detected."
    )


# ============================================================
# 12. RANGE VALIDATION
# ============================================================

# Rates must be between 0 and 1

rate_columns = [
    "baseline_win_rate",
    "baseline_scoring_match_rate",
    "baseline_clean_sheet_rate"
]

for column in rate_columns:

    invalid = (
        (baseline_df[column] < 0)
        | (baseline_df[column] > 1)
    )

    if invalid.any():
        raise ValueError(
            f"Invalid values detected in {column}."
        )


# Standard deviation cannot be negative

if (
    baseline_df[
        "baseline_scoring_consistency"
    ] < 0
).any():

    raise ValueError(
        "Negative scoring consistency detected."
    )


print("\nBaseline range validation: PASSED")


# ============================================================
# 13. DISPLAY EXAMPLE BASELINES
# ============================================================

display_columns = [
    "team",
    "baseline_matches",

    "baseline_avg_goals_scored",
    "baseline_avg_goals_conceded",

    "baseline_points_per_match",

    "baseline_scoring_consistency",
    "baseline_scoring_match_rate",
    "baseline_clean_sheet_rate"
]

print(
    "\nExample team baselines:"
)

print(
    baseline_df[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 14. SAVE
# ============================================================

baseline_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("PRE-TOURNAMENT BASELINE FILE SAVED")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print(
    "\nBaseline features created:"
)

for column in baseline_columns:
    print(" -", column)

print(
    "\nIMPORTANT: These baselines use only "
    "the selected pre-tournament match data."
)

print(
    "\nPre-tournament baseline generation "
    "completed successfully."
)