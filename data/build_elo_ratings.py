import os
import math
import pandas as pd


# ============================================================
# HIT140 ASSESSMENT 3
# PRE-MATCH ELO RATING BUILDER
#
# Purpose:
# Create a pre-match Elo rating for every team in every
# FIFA World Cup 2026 match.
#
# IMPORTANT:
# - Ratings are updated chronologically.
# - A match can only affect Elo AFTER that match.
# - Therefore the Elo value attached to a World Cup match
#   represents information available BEFORE that match.
#
# Output:
# world_cup_2026_prematch_elo.csv
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

HISTORY_FILE = os.path.join(
    DATA_DIR,
    "international_results.csv"
)

MASTER_FILE = os.path.join(
    DATA_DIR,
    "world_cup_2026_master_matches.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "world_cup_2026_prematch_elo.csv"
)


# ============================================================
# 2. ELO SETTINGS
# ============================================================
#
# All teams begin at the same neutral rating.
#
# K controls how strongly one result changes the rating.
# We use one fixed K throughout so the calculation remains
# transparent and reproducible.
# ============================================================

INITIAL_ELO = 1500.0
K_FACTOR = 20.0


# ============================================================
# 3. TEAM NAME STANDARDISATION
# ============================================================
#
# international_results.csv and FIFA data can use slightly
# different team names.
#
# Add aliases here only when needed.
# ============================================================

TEAM_ALIASES = {
    "United States": "USA",
    "United States of America": "USA",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Czech Republic": "Czechia",
    "Türkiye": "Türkiye",
    "Turkey": "Türkiye",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR"
}


def standardise_team_name(name):

    if pd.isna(name):
        return name

    name = str(name).strip()

    return TEAM_ALIASES.get(
        name,
        name
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 72)
print("HIT140 ASSESSMENT 3 - PRE-MATCH ELO RATING BUILDER")
print("=" * 72)


print("\nLoading historical international results...")

history_df = pd.read_csv(
    HISTORY_FILE
)

print(
    "Historical matches loaded:",
    len(history_df)
)


print("\nLoading World Cup master database...")

master_df = pd.read_csv(
    MASTER_FILE
)

print(
    "World Cup matches loaded:",
    len(master_df)
)


# ============================================================
# 5. IDENTIFY HISTORICAL DATA COLUMN NAMES
# ============================================================
#
# Your international_results.csv is expected to contain
# something equivalent to:
#
# date
# home_team
# away_team
# home_score
# away_score
#
# This section detects common alternatives.
# ============================================================

print("\nHistorical columns:")

print(
    history_df.columns.tolist()
)


def find_column(
    dataframe,
    possible_names
):

    for column in possible_names:

        if column in dataframe.columns:
            return column

    return None


history_date_column = find_column(
    history_df,
    [
        "date",
        "match_date"
    ]
)

history_home_team_column = find_column(
    history_df,
    [
        "home_team",
        "home"
    ]
)

history_away_team_column = find_column(
    history_df,
    [
        "away_team",
        "away"
    ]
)

history_home_score_column = find_column(
    history_df,
    [
        "home_score",
        "home_goals"
    ]
)

history_away_score_column = find_column(
    history_df,
    [
        "away_score",
        "away_goals"
    ]
)


required_detected_columns = {
    "date": history_date_column,
    "home team": history_home_team_column,
    "away team": history_away_team_column,
    "home score": history_home_score_column,
    "away score": history_away_score_column
}


missing_detected_columns = [
    name
    for name, column
    in required_detected_columns.items()
    if column is None
]


if missing_detected_columns:

    raise ValueError(
        "Could not identify these historical columns: "
        f"{missing_detected_columns}"
    )


print("\nDetected historical columns:")

for name, column in required_detected_columns.items():

    print(
        f"{name}: {column}"
    )


# ============================================================
# 6. STANDARDISE HISTORICAL DATA
# ============================================================

history = history_df[
    [
        history_date_column,
        history_home_team_column,
        history_away_team_column,
        history_home_score_column,
        history_away_score_column
    ]
].copy()


history.columns = [
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals"
]


history["match_date"] = pd.to_datetime(
    history["match_date"],
    errors="coerce",
    utc=True
)


history["home_team"] = (
    history["home_team"]
    .apply(standardise_team_name)
)

history["away_team"] = (
    history["away_team"]
    .apply(standardise_team_name)
)


history["home_goals"] = pd.to_numeric(
    history["home_goals"],
    errors="coerce"
)

history["away_goals"] = pd.to_numeric(
    history["away_goals"],
    errors="coerce"
)


# ============================================================
# 7. REMOVE INVALID HISTORICAL ROWS
# ============================================================

before_cleaning = len(history)


history = history.dropna(
    subset=[
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals"
    ]
).copy()


after_cleaning = len(history)


print(
    "\nHistorical rows removed because of missing data:",
    before_cleaning - after_cleaning
)

print(
    "Usable historical matches:",
    after_cleaning
)


# ============================================================
# 8. PREPARE WORLD CUP DATABASE
# ============================================================

master_required = [
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals"
]


missing_master_columns = [
    column
    for column in master_required
    if column not in master_df.columns
]


if missing_master_columns:

    raise ValueError(
        "Master database missing required columns: "
        f"{missing_master_columns}"
    )


if len(master_df) != 104:

    raise ValueError(
        "World Cup master database must contain exactly "
        f"104 matches. Found {len(master_df)}."
    )


if master_df["match_id"].nunique() != 104:

    raise ValueError(
        "World Cup master database must contain "
        "104 unique match IDs."
    )


master = master_df[
    master_required
].copy()


master["match_date"] = pd.to_datetime(
    master["match_date"],
    errors="coerce",
    utc=True
)


if master["match_date"].isna().any():

    raise ValueError(
        "Some World Cup match dates could not be converted."
    )


master["home_team"] = (
    master["home_team"]
    .apply(standardise_team_name)
)

master["away_team"] = (
    master["away_team"]
    .apply(standardise_team_name)
)


master["home_goals"] = pd.to_numeric(
    master["home_goals"],
    errors="coerce"
)

master["away_goals"] = pd.to_numeric(
    master["away_goals"],
    errors="coerce"
)


if master[
    [
        "home_goals",
        "away_goals"
    ]
].isna().any().any():

    raise ValueError(
        "Some World Cup scores are missing."
    )


master = master.sort_values(
    [
        "match_date",
        "match_id"
    ]
).reset_index(drop=True)


print(
    "\nWorld Cup master database validation: PASSED"
)


# ============================================================
# 9. IDENTIFY WORLD CUP TEAMS
# ============================================================

world_cup_teams = sorted(
    set(master["home_team"])
    |
    set(master["away_team"])
)


print(
    "World Cup teams found:",
    len(world_cup_teams)
)


if len(world_cup_teams) != 48:

    print(
        "\nWARNING: Expected 48 World Cup teams, "
        f"but found {len(world_cup_teams)}."
    )


# ============================================================
# 10. CHECK HISTORICAL TEAM COVERAGE
# ============================================================

historical_teams = (
    set(history["home_team"])
    |
    set(history["away_team"])
)


missing_history_teams = sorted(
    set(world_cup_teams)
    - historical_teams
)


print(
    "\nWorld Cup teams missing completely "
    "from historical database:"
)


if missing_history_teams:

    for team in missing_history_teams:
        print(" -", team)

else:
    print("None")


# ============================================================
# 11. REMOVE HISTORICAL MATCHES THAT OCCUR ON OR AFTER
#     THE FIRST WORLD CUP MATCH
# ============================================================
#
# We build a historical Elo state first.
#
# To prevent future World Cup information from entering the
# initial ratings, historical data must end before the first
# World Cup match in the master database.
# ============================================================

first_world_cup_datetime = (
    master["match_date"].min()
)


print(
    "\nFirst World Cup match datetime:",
    first_world_cup_datetime
)


historical_for_elo = history[
    history["match_date"]
    < first_world_cup_datetime
].copy()


historical_for_elo = (
    historical_for_elo
    .sort_values(
        [
            "match_date",
            "home_team",
            "away_team"
        ]
    )
    .reset_index(drop=True)
)


print(
    "Historical matches before first World Cup match:",
    len(historical_for_elo)
)


if len(historical_for_elo) == 0:

    raise ValueError(
        "No historical matches exist before "
        "the first World Cup match."
    )


# ============================================================
# 12. ELO FUNCTIONS
# ============================================================


def expected_score(
    rating_a,
    rating_b
):

    return 1.0 / (
        1.0
        + math.pow(
            10.0,
            (rating_b - rating_a) / 400.0
        )
    )


def actual_result(
    goals_a,
    goals_b
):

    if goals_a > goals_b:
        return 1.0

    if goals_a < goals_b:
        return 0.0

    return 0.5


def update_elo(
    home_rating,
    away_rating,
    home_goals,
    away_goals
):

    home_expected = expected_score(
        home_rating,
        away_rating
    )

    away_expected = expected_score(
        away_rating,
        home_rating
    )


    home_actual = actual_result(
        home_goals,
        away_goals
    )

    away_actual = 1.0 - home_actual


    new_home_rating = (
        home_rating
        + K_FACTOR
        * (
            home_actual
            - home_expected
        )
    )


    new_away_rating = (
        away_rating
        + K_FACTOR
        * (
            away_actual
            - away_expected
        )
    )


    return (
        new_home_rating,
        new_away_rating
    )


# ============================================================
# 13. INITIALISE ELO RATINGS
# ============================================================

all_teams = (
    set(historical_for_elo["home_team"])
    |
    set(historical_for_elo["away_team"])
    |
    set(world_cup_teams)
)


elo_ratings = {
    team: INITIAL_ELO
    for team in all_teams
}


matches_processed = {
    team: 0
    for team in all_teams
}


print(
    "\nTeams initialised in Elo system:",
    len(elo_ratings)
)


# ============================================================
# 14. PROCESS HISTORICAL MATCHES CHRONOLOGICALLY
# ============================================================

print(
    "\nCalculating historical Elo ratings..."
)


for row in historical_for_elo.itertuples(
    index=False
):

    home_team = row.home_team
    away_team = row.away_team

    home_goals = row.home_goals
    away_goals = row.away_goals


    home_rating = elo_ratings.get(
        home_team,
        INITIAL_ELO
    )

    away_rating = elo_ratings.get(
        away_team,
        INITIAL_ELO
    )


    new_home_rating, new_away_rating = update_elo(
        home_rating,
        away_rating,
        home_goals,
        away_goals
    )


    elo_ratings[home_team] = (
        new_home_rating
    )

    elo_ratings[away_team] = (
        new_away_rating
    )


    matches_processed[home_team] = (
        matches_processed.get(
            home_team,
            0
        )
        + 1
    )

    matches_processed[away_team] = (
        matches_processed.get(
            away_team,
            0
        )
        + 1
    )


print(
    "Historical Elo calculation: COMPLETE"
)


# ============================================================
# 15. DISPLAY PRE-TOURNAMENT WORLD CUP ELO RATINGS
# ============================================================

pretournament_rows = []


for team in world_cup_teams:

    pretournament_rows.append(
        {
            "team": team,
            "pre_tournament_elo":
                elo_ratings.get(
                    team,
                    INITIAL_ELO
                ),
            "historical_matches_used":
                matches_processed.get(
                    team,
                    0
                )
        }
    )


pretournament_df = pd.DataFrame(
    pretournament_rows
)


pretournament_df = (
    pretournament_df
    .sort_values(
        "pre_tournament_elo",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nTop pre-tournament Elo ratings:"
)

print(
    pretournament_df
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. CREATE WORLD CUP PRE-MATCH ELO VALUES
# ============================================================
#
# CRITICAL ORDER:
#
# 1. Read Elo BEFORE the World Cup match.
# 2. Store those values.
# 3. Only then update Elo using that match result.
#
# This prevents target leakage.
# ============================================================

print(
    "\nCreating World Cup pre-match Elo observations..."
)


world_cup_elo_rows = []


for row in master.itertuples(
    index=False
):

    match_id = row.match_id
    match_date = row.match_date

    home_team = row.home_team
    away_team = row.away_team

    home_goals = row.home_goals
    away_goals = row.away_goals


    # --------------------------------------------------------
    # Elo BEFORE this match
    # --------------------------------------------------------

    home_elo_before = elo_ratings.get(
        home_team,
        INITIAL_ELO
    )

    away_elo_before = elo_ratings.get(
        away_team,
        INITIAL_ELO
    )


    elo_difference = (
        home_elo_before
        - away_elo_before
    )


    # --------------------------------------------------------
    # Save PRE-MATCH values
    # --------------------------------------------------------

    world_cup_elo_rows.append(
        {
            "match_id":
                match_id,

            "match_date":
                match_date,

            "home_team":
                home_team,

            "away_team":
                away_team,

            "home_elo_before":
                home_elo_before,

            "away_elo_before":
                away_elo_before,

            "elo_rating_difference":
                elo_difference
        }
    )


    # --------------------------------------------------------
    # Update Elo only AFTER pre-match values are saved
    # --------------------------------------------------------

    new_home_elo, new_away_elo = update_elo(
        home_elo_before,
        away_elo_before,
        home_goals,
        away_goals
    )


    elo_ratings[home_team] = (
        new_home_elo
    )

    elo_ratings[away_team] = (
        new_away_elo
    )


world_cup_elo_df = pd.DataFrame(
    world_cup_elo_rows
)


# ============================================================
# 17. STRUCTURAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "WORLD CUP ELO VALIDATION"
)

print(
    "=" * 72
)


print(
    "\nRows:",
    len(world_cup_elo_df)
)

print(
    "Unique matches:",
    world_cup_elo_df[
        "match_id"
    ].nunique()
)


duplicate_matches = (
    world_cup_elo_df[
        "match_id"
    ]
    .duplicated()
    .sum()
)


print(
    "Duplicate match IDs:",
    duplicate_matches
)


if len(world_cup_elo_df) != 104:

    raise ValueError(
        "Elo output must contain exactly "
        f"104 rows. Found {len(world_cup_elo_df)}."
    )


if (
    world_cup_elo_df[
        "match_id"
    ].nunique()
    != 104
):

    raise ValueError(
        "Elo output must contain "
        "104 unique matches."
    )


if duplicate_matches != 0:

    raise ValueError(
        "Duplicate match IDs detected "
        "in Elo output."
    )


print(
    "\nStructure validation: PASSED"
)


# ============================================================
# 18. MISSING VALUE VALIDATION
# ============================================================

elo_columns = [
    "home_elo_before",
    "away_elo_before",
    "elo_rating_difference"
]


print(
    "\nMissing Elo values:"
)


missing_elo = (
    world_cup_elo_df[
        elo_columns
    ]
    .isna()
    .sum()
)


print(
    missing_elo
)


if missing_elo.sum() != 0:

    raise ValueError(
        "Missing Elo values detected."
    )


print(
    "Missing-value validation: PASSED"
)


# ============================================================
# 19. VARIATION CHECK
# ============================================================

print(
    "\nElo difference summary:"
)


print(
    world_cup_elo_df[
        "elo_rating_difference"
    ]
    .describe()
)


unique_elo_differences = (
    world_cup_elo_df[
        "elo_rating_difference"
    ]
    .nunique()
)


print(
    "\nUnique Elo differences:",
    unique_elo_differences
)


if unique_elo_differences <= 1:

    raise ValueError(
        "Elo difference has no useful variation."
    )


# ============================================================
# 20. CHECK WORLD CUP TEAMS' HISTORICAL COVERAGE
# ============================================================

coverage_rows = []


for team in world_cup_teams:

    coverage_rows.append(
        {
            "team":
                team,

            "historical_matches_used":
                matches_processed.get(
                    team,
                    0
                )
        }
    )


coverage_df = pd.DataFrame(
    coverage_rows
)


coverage_df = (
    coverage_df
    .sort_values(
        "historical_matches_used"
    )
    .reset_index(drop=True)
)


print(
    "\nHistorical-match coverage for World Cup teams:"
)


print(
    coverage_df.to_string(
        index=False
    )
)


zero_history = coverage_df[
    coverage_df[
        "historical_matches_used"
    ] == 0
]


print(
    "\nWorld Cup teams with zero "
    "historical matches used:",
    len(zero_history)
)


if len(zero_history) > 0:

    print(
        zero_history.to_string(
            index=False
        )
    )


# ============================================================
# 21. DISPLAY SAMPLE
# ============================================================

print(
    "\nExample World Cup pre-match Elo observations:"
)


print(
    world_cup_elo_df
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE OUTPUT
# ============================================================

world_cup_elo_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n" + "=" * 72
)

print(
    "PRE-MATCH ELO FILE SAVED"
)

print(
    "=" * 72
)


print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


print(
    "\nColumns created:"
)

print(
    " - match_id"
)

print(
    " - match_date"
)

print(
    " - home_team"
)

print(
    " - away_team"
)

print(
    " - home_elo_before"
)

print(
    " - away_elo_before"
)

print(
    " - elo_rating_difference"
)


print(
    "\nIMPORTANT:"
)

print(
    "Each World Cup Elo value was recorded BEFORE "
    "that match result was used to update the ratings."
)

print(
    "Therefore the World Cup target match itself "
    "does not leak into its Elo predictor."
)