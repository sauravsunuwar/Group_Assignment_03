import os
import pandas as pd


# ============================================================
# HIT140 ASSESSMENT 3
# PRE-MATCH DATA INTEGRITY / LEAKAGE AUDIT
# ============================================================
#
# Main purpose:
# Check whether the historical matches selected for each team's
# pre-tournament baseline occurred BEFORE that team's first
# FIFA World Cup 2026 match.
#
# This helps detect future-information leakage.
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def find_file(filename):

    possible_paths = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "data", filename)
    ]

    for path in possible_paths:

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"\nCould not find: {filename}\n"
        f"Checked:\n"
        + "\n".join(possible_paths)
    )


MASTER_FILE = find_file(
    "world_cup_2026_master_matches.csv"
)

BASELINE_MATCH_FILE = find_file(
    "pre_tournament_matches.csv"
)

BASELINE_FILE = find_file(
    "pre_tournament_baselines.csv"
)

PREMATCH_FEATURE_FILE = find_file(
    "world_cup_2026_team_prematch_features.csv"
)


# ============================================================
# 2. LOAD FILES
# ============================================================

print("=" * 76)
print("HIT140 - PRE-MATCH DATA INTEGRITY AUDIT")
print("=" * 76)


master = pd.read_csv(
    MASTER_FILE
)

baseline_matches = pd.read_csv(
    BASELINE_MATCH_FILE
)

baselines = pd.read_csv(
    BASELINE_FILE
)

features = pd.read_csv(
    PREMATCH_FEATURE_FILE
)


print("\nFiles loaded:")

print(
    "Master matches:",
    len(master)
)

print(
    "Pre-tournament match rows:",
    len(baseline_matches)
)

print(
    "Pre-tournament baseline teams:",
    len(baselines)
)

print(
    "Pre-match feature rows:",
    len(features)
)


# ============================================================
# 3. SHOW AVAILABLE BASELINE-MATCH COLUMNS
# ============================================================

print(
    "\nColumns in pre_tournament_matches.csv:"
)

for column in baseline_matches.columns:

    print(
        " -",
        column
    )


# ============================================================
# 4. DETECT IMPORTANT COLUMN NAMES
# ============================================================

def find_column(
    dataframe,
    candidates,
    description
):

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    raise ValueError(
        f"\nCould not identify {description} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


baseline_team_column = find_column(
    baseline_matches,
    [
        "team",
        "team_name"
    ],
    "baseline team"
)


baseline_date_column = find_column(
    baseline_matches,
    [
        "date",
        "match_date"
    ],
    "baseline match date"
)


print(
    "\nDetected baseline team column:",
    baseline_team_column
)

print(
    "Detected baseline date column:",
    baseline_date_column
)


# ============================================================
# 5. CONVERT DATES
# ============================================================
# ============================================================
# CONVERT ALL DATES TO UTC
# ============================================================
#
# Some World Cup timestamps contain timezone information,
# while historical match dates may not.
#
# utc=True standardises both sources so they can be compared
# safely without tz-aware / tz-naive errors.
# ============================================================

master[
    "match_date"
] = pd.to_datetime(
    master[
        "match_date"
    ],
    errors="coerce",
    utc=True
)


baseline_matches[
    baseline_date_column
] = pd.to_datetime(
    baseline_matches[
        baseline_date_column
    ],
    errors="coerce",
    utc=True
)
# ============================================================
# 6. BUILD FIRST WORLD CUP MATCH DATE FOR EACH TEAM
# ============================================================

home_dates = master[
    [
        "home_team",
        "match_date"
    ]
].copy()


home_dates = home_dates.rename(
    columns={
        "home_team":
            "team"
    }
)


away_dates = master[
    [
        "away_team",
        "match_date"
    ]
].copy()


away_dates = away_dates.rename(
    columns={
        "away_team":
            "team"
    }
)


team_world_cup_dates = pd.concat(
    [
        home_dates,
        away_dates
    ],
    ignore_index=True
)


first_world_cup_dates = (
    team_world_cup_dates
    .groupby(
        "team",
        as_index=False
    )[
        "match_date"
    ]
    .min()
    .rename(
        columns={
            "match_date":
                "first_world_cup_match_date"
        }
    )
)


print(
    "\nWorld Cup teams found:",
    len(first_world_cup_dates)
)


if len(first_world_cup_dates) != 48:

    raise ValueError(
        f"Expected 48 World Cup teams, "
        f"found {len(first_world_cup_dates)}."
    )


print(
    "First World Cup match-date construction: PASSED"
)


# ============================================================
# 7. CHECK TEAM-NAME MATCHING
# ============================================================

baseline_team_names = set(
    baseline_matches[
        baseline_team_column
    ].dropna()
)


world_cup_team_names = set(
    first_world_cup_dates[
        "team"
    ].dropna()
)


missing_baseline_teams = sorted(
    world_cup_team_names
    -
    baseline_team_names
)


extra_baseline_teams = sorted(
    baseline_team_names
    -
    world_cup_team_names
)


print(
    "\nWorld Cup teams missing "
    "from baseline match data:"
)


if missing_baseline_teams:

    for team in missing_baseline_teams:

        print(
            " -",
            team
        )

else:

    print(
        "None"
    )


print(
    "\nExtra teams in baseline match data:"
)


if extra_baseline_teams:

    for team in extra_baseline_teams:

        print(
            " -",
            team
        )

else:

    print(
        "None"
    )


if missing_baseline_teams:

    raise ValueError(
        "Some World Cup teams have no "
        "baseline match data."
    )


# ============================================================
# 8. MERGE FIRST WORLD CUP MATCH DATES
# ============================================================

audit = baseline_matches.merge(
    first_world_cup_dates,
    left_on=baseline_team_column,
    right_on="team",
    how="left",
    validate="many_to_one"
)


missing_first_dates = (
    audit[
        "first_world_cup_match_date"
    ]
    .isna()
    .sum()
)


print(
    "\nBaseline rows without matched "
    "World Cup first-match date:",
    missing_first_dates
)


if missing_first_dates != 0:

    bad_rows = audit[
        audit[
            "first_world_cup_match_date"
        ].isna()
    ]

    print(
        bad_rows.to_string(
            index=False
        )
    )

    raise ValueError(
        "Team-name matching problem detected."
    )


# ============================================================
# 9. CRITICAL LEAKAGE CHECK
# ============================================================
# Historical baseline match must be on an EARLIER CALENDAR DATE
# than the team's first World Cup match.
#
# We deliberately compare calendar dates rather than timestamps
# because the historical results source contains date-only values
# that pandas represents as 00:00:00.

audit[
    "valid_pre_tournament_match"
] = (
    audit[
        baseline_date_column
    ].dt.date
    <
    audit[
        "first_world_cup_match_date"
    ].dt.date
)

invalid_rows = audit[
    ~audit[
        "valid_pre_tournament_match"
    ]
].copy()


print(
    "\n" + "=" * 76
)

print(
    "CRITICAL BASELINE DATE CHECK"
)

print(
    "=" * 76
)


print(
    "\nTotal baseline team-match rows:",
    len(audit)
)


print(
    "Valid rows occurring before "
    "team's first World Cup match:",
    int(
        audit[
            "valid_pre_tournament_match"
        ].sum()
    )
)


print(
    "Suspicious/invalid rows:",
    len(invalid_rows)
)


if len(invalid_rows) == 0:

    print(
        "\nPASS:"
    )

    print(
        "Every selected baseline match occurred "
        "before that team's first World Cup match."
    )

else:

    print(
        "\nFAIL:"
    )

    print(
        "The following baseline rows occurred "
        "on or after the team's first World Cup match."
    )


    display_columns = [
        baseline_team_column,
        baseline_date_column,
        "first_world_cup_match_date"
    ]


    # Add useful descriptive columns if available.

    for column in [
        "home_team",
        "away_team",
        "opponent",
        "goals_scored",
        "goals_conceded"
    ]:

        if (
            column in invalid_rows.columns
            and
            column not in display_columns
        ):

            display_columns.append(
                column
            )


    print(
        "\n" +
        invalid_rows[
            display_columns
        ]
        .sort_values(
            [
                baseline_team_column,
                baseline_date_column
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 10. EXACTLY FIVE BASELINE MATCHES PER TEAM
# ============================================================

matches_per_team = (
    audit
    .groupby(
        baseline_team_column
    )
    .size()
)


print(
    "\n" + "=" * 76
)

print(
    "BASELINE MATCH COUNT CHECK"
)

print(
    "=" * 76
)


print(
    "\nMinimum matches per team:",
    matches_per_team.min()
)


print(
    "Maximum matches per team:",
    matches_per_team.max()
)


print(
    "Teams represented:",
    len(matches_per_team)
)


incorrect_counts = (
    matches_per_team[
        matches_per_team != 5
    ]
)


print(
    "Teams without exactly 5 "
    "baseline matches:",
    len(incorrect_counts)
)


if len(incorrect_counts) > 0:

    print(
        "\nTeams with incorrect counts:"
    )

    print(
        incorrect_counts.to_string()
    )

else:

    print(
        "Baseline match-count validation: PASSED"
    )


# ============================================================
# 11. LATEST BASELINE DATE FOR EACH TEAM
# ============================================================

latest_baseline = (
    audit
    .groupby(
        baseline_team_column,
        as_index=False
    )
    .agg(
        latest_baseline_match=(
            baseline_date_column,
            "max"
        ),

        first_world_cup_match=(
            "first_world_cup_match_date",
            "first"
        )
    )
)


latest_baseline[
    "days_before_first_world_cup_match"
] = (
    latest_baseline[
        "first_world_cup_match"
    ]
    -
    latest_baseline[
        "latest_baseline_match"
    ]
).dt.days


print(
    "\n" + "=" * 76
)

print(
    "LATEST BASELINE MATCH BY TEAM"
)

print(
    "=" * 76
)


print(
    latest_baseline
    .sort_values(
        "days_before_first_world_cup_match"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 12. SAME-DAY BASELINE CHECK
# ============================================================

same_day = latest_baseline[
    latest_baseline[
        "days_before_first_world_cup_match"
    ]
    <= 0
]


print(
    "\nTeams whose latest baseline match "
    "was on/after first World Cup match:",
    len(same_day)
)


if len(same_day) > 0:

    print(
        same_day.to_string(
            index=False
        )
    )


# ============================================================
# 13. BASELINE TEAM COVERAGE
# ============================================================

print(
    "\n" + "=" * 76
)

print(
    "BASELINE SUMMARY FILE CHECK"
)

print(
    "=" * 76
)


print(
    "\nRows in pre_tournament_baselines.csv:",
    len(baselines)
)


if "team" in baselines.columns:

    print(
        "Unique baseline teams:",
        baselines[
            "team"
        ].nunique()
    )


    if (
        baselines[
            "team"
        ].nunique()
        == 48
    ):

        print(
            "Baseline team coverage: PASSED"
        )

    else:

        print(
            "WARNING: Expected 48 baseline teams."
        )


# ============================================================
# 14. PRE-MATCH FEATURE DATASET STRUCTURE
# ============================================================

print(
    "\n" + "=" * 76
)

print(
    "PRE-MATCH FEATURE DATASET CHECK"
)

print(
    "=" * 76
)


print(
    "\nRows:",
    len(features)
)


if "match_id" in features.columns:

    print(
        "Unique matches:",
        features[
            "match_id"
        ].nunique()
    )


if "team" in features.columns:

    print(
        "Unique teams:",
        features[
            "team"
        ].nunique()
    )


if (
    "match_id" in features.columns
    and
    "team" in features.columns
):

    duplicate_team_matches = (
        features
        .duplicated(
            subset=[
                "match_id",
                "team"
            ]
        )
        .sum()
    )


    print(
        "Duplicate team-match rows:",
        duplicate_team_matches
    )


if len(features) != 208:

    print(
        "WARNING: Expected 208 "
        "team-match observations."
    )

else:

    print(
        "208-row structure: PASSED"
    )


# ============================================================
# 15. FIRST-TOURNAMENT-MATCH FEATURE CHECK
# ============================================================

if (
    "first_tournament_match"
    in features.columns
):

    first_rows = features[
        features[
            "first_tournament_match"
        ]
        == 1
    ]


    print(
        "\nFirst tournament match rows:",
        len(first_rows)
    )


    if len(first_rows) == 48:

        print(
            "First-match team coverage: PASSED"
        )

    else:

        print(
            "WARNING: Expected exactly "
            "48 first-match rows."
        )


# ============================================================
# 16. FINAL AUDIT RESULT
# ============================================================

print(
    "\n" + "=" * 76
)

print(
    "FINAL PRE-MATCH INTEGRITY RESULT"
)

print(
    "=" * 76
)


audit_passed = True


if len(invalid_rows) != 0:

    audit_passed = False


if len(incorrect_counts) != 0:

    audit_passed = False


if len(matches_per_team) != 48:

    audit_passed = False


if len(features) != 208:

    audit_passed = False


if audit_passed:

    print(
        "\nPASS"
    )

    print(
        "The automated checks found no "
        "baseline-date leakage."
    )

    print(
        "All 48 teams have exactly five "
        "selected baseline matches occurring "
        "before their first World Cup match."
    )

else:

    print(
        "\nFAIL / REVIEW REQUIRED"
    )

    print(
        "At least one integrity check failed."
    )

    print(
        "Do NOT treat the pre-match datasets "
        "as final until the flagged rows "
        "have been investigated."
    )


print(
    "\nNOTE:"
)

print(
    "This script verifies chronological integrity."
)

print(
    "It does not by itself prove that every historical "
    "source record was genuinely available/public "
    "on that historical date."
)