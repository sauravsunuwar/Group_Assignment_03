import os
import pandas as pd


# ============================================================
# HIT140 ASSESSMENT 3
# REGRESSION 2.2 DATASET BUILDER
# ============================================================
#
# Target:
# goals_scored
#
# Requirements:
# - Exactly 208 team-match observations
# - Exactly 8 explanatory variables
# - All predictors must contain information available
#   BEFORE the target match
#
# Final predictors:
# 1. avg_goals_scored_before
# 2. opponent_avg_goals_conceded_before
# 3. team_fifa_points
# 4. opponent_fifa_points
# 5. win_rate_before
# 6. opponent_scoring_match_rate_before
# 7. avg_goal_difference_before
# 8. opponent_clean_sheet_rate_before
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


FEATURE_FILE = find_file(
    "world_cup_2026_team_prematch_features.csv"
)

OUTPUT_FILE = os.path.join(
    os.path.dirname(FEATURE_FILE),
    "regression_2_2_dataset.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 72)
print("HIT140 ASSESSMENT 3 - REGRESSION 2.2 DATASET")
print("=" * 72)

print("\nLoading pre-match team feature database...")

df = pd.read_csv(FEATURE_FILE)

print("Rows loaded:", len(df))


# ============================================================
# 3. DEFINE TARGET AND EXACTLY 8 PREDICTORS
# ============================================================

target = "goals_scored"


predictors = [

    "avg_goals_scored_before",

    "opponent_avg_goals_conceded_before",

    "team_fifa_points",

    "opponent_fifa_points",

    "win_rate_before",

    "opponent_scoring_match_rate_before",

    "avg_goal_difference_before",

    "opponent_clean_sheet_rate_before"

]

print("\nTarget:")
print(target)


print("\nCandidate explanatory variables:")

for number, predictor in enumerate(
    predictors,
    start=1
):

    print(
        f"{number}. {predictor}"
    )


if len(predictors) != 8:

    raise ValueError(
        "Regression 2.2 requires exactly "
        "8 explanatory variables."
    )


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

identifier_columns = [

    "match_id",

    "match_date",

    "stage",

    "team_id",

    "team",

    "opponent_id",

    "opponent",

    "home_away"
]


required_columns = (
    identifier_columns
    +
    [target]
    +
    predictors
)


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


print(
    "\nRequired-column validation: PASSED"
)


# ============================================================
# 5. CREATE REGRESSION DATASET
# ============================================================

regression_df = df[
    required_columns
].copy()


# ============================================================
# 6. STRUCTURAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "REGRESSION 2.2 STRUCTURAL VALIDATION"
)

print(
    "=" * 72
)


print(
    "\nRows:",
    len(regression_df)
)


print(
    "Unique matches:",
    regression_df[
        "match_id"
    ].nunique()
)


print(
    "Unique teams:",
    regression_df[
        "team"
    ].nunique()
)


duplicate_team_matches = (
    regression_df
    .duplicated(
        subset=[
            "match_id",
            "team_id"
        ]
    )
    .sum()
)


print(
    "Duplicate team-match rows:",
    duplicate_team_matches
)


print(
    "Number of explanatory variables:",
    len(predictors)
)


if len(regression_df) != 208:

    raise ValueError(
        f"Regression 2.2 must contain "
        f"exactly 208 rows. "
        f"Found {len(regression_df)}."
    )


if (
    regression_df[
        "match_id"
    ]
    .nunique()
    != 104
):

    raise ValueError(
        "Regression 2.2 should represent "
        "104 unique matches."
    )


if duplicate_team_matches != 0:

    raise ValueError(
        "Duplicate team-match observations "
        "were detected."
    )


if len(predictors) != 8:

    raise ValueError(
        "Exactly 8 predictors are required."
    )


print(
    "\nStructural validation: PASSED"
)


# ============================================================
# 7. VERIFY TWO OBSERVATIONS PER MATCH
# ============================================================

rows_per_match = (
    regression_df
    .groupby(
        "match_id"
    )
    .size()
)


incorrect_matches = (
    rows_per_match[
        rows_per_match != 2
    ]
)


print(
    "\nMatches without exactly "
    "2 team observations:",
    len(incorrect_matches)
)


if len(incorrect_matches) != 0:

    print(
        incorrect_matches
    )

    raise ValueError(
        "Every match must have exactly "
        "two team observations."
    )


print(
    "Two-team-per-match validation: PASSED"
)


# ============================================================
# 8. MISSING VALUE CHECK
# ============================================================

missing_values = (
    regression_df[
        predictors
    ]
    .isna()
    .sum()
)


print(
    "\nMissing values in predictors:"
)


print(
    missing_values
)


rows_with_missing = (
    regression_df[
        predictors
    ]
    .isna()
    .any(axis=1)
    .sum()
)


print(
    "\nRows with at least one "
    "missing predictor:",
    rows_with_missing
)


if rows_with_missing != 0:

    raise ValueError(
        "Missing predictor values detected."
    )


if (
    regression_df[
        target
    ]
    .isna()
    .sum()
    != 0
):

    raise ValueError(
        "Missing target values detected."
    )


print(
    "Missing-value validation: PASSED"
)


# ============================================================
# 9. CONSTANT VARIABLE CHECK
# ============================================================

unique_counts = (
    regression_df[
        predictors
    ]
    .nunique()
)


print(
    "\nPredictor unique-value counts:"
)


print(
    unique_counts
)


constant_predictors = (
    unique_counts[
        unique_counts <= 1
    ]
    .index
    .tolist()
)


if constant_predictors:

    raise ValueError(
        f"Constant predictors detected: "
        f"{constant_predictors}"
    )


print(
    "\nNo constant predictors detected."
)


# ============================================================
# 10. EXACT REDUNDANCY CHECK
# ============================================================

print(
    "\nChecking exact predictor redundancy..."
)


redundant_pairs = []


for i in range(
    len(predictors)
):

    for j in range(
        i + 1,
        len(predictors)
    ):

        predictor_a = predictors[i]

        predictor_b = predictors[j]


        same = (
            regression_df[
                predictor_a
            ]
            .equals(
                regression_df[
                    predictor_b
                ]
            )
        )


        opposite = (
            regression_df[
                predictor_a
            ]
            .equals(
                -regression_df[
                    predictor_b
                ]
            )
        )


        if same or opposite:

            redundant_pairs.append(
                (
                    predictor_a,
                    predictor_b
                )
            )


if redundant_pairs:

    print(
        "Exact redundant predictor pairs:"
    )

    for pair in redundant_pairs:

        print(
            " -",
            pair
        )

else:

    print(
        "No exact duplicate/opposite "
        "predictors detected."
    )


# ============================================================
# 11. CORRELATION MATRIX
# ============================================================

correlation_matrix = (
    regression_df[
        predictors
    ]
    .corr()
)


print(
    "\n" + "=" * 72
)

print(
    "PREDICTOR CORRELATION MATRIX"
)

print(
    "=" * 72
)


print(
    correlation_matrix
    .round(3)
    .to_string()
)


# ============================================================
# 12. HIGH CORRELATION CHECK
# ============================================================

threshold = 0.80

high_correlation_pairs = []


for i in range(
    len(predictors)
):

    for j in range(
        i + 1,
        len(predictors)
    ):

        predictor_a = predictors[i]

        predictor_b = predictors[j]


        correlation = (
            correlation_matrix.loc[
                predictor_a,
                predictor_b
            ]
        )


        if abs(correlation) >= threshold:

            high_correlation_pairs.append(
                (
                    predictor_a,
                    predictor_b,
                    correlation
                )
            )


print(
    "\nPredictor pairs with "
    "|correlation| >= 0.80:"
)


if high_correlation_pairs:

    for (
        predictor_a,
        predictor_b,
        correlation
    ) in high_correlation_pairs:

        print(
            f" - {predictor_a} "
            f"<-> {predictor_b}: "
            f"{correlation:.3f}"
        )

else:

    print(
        "None"
    )


# ============================================================
# 13. CORRELATION WITH TARGET
# ============================================================

target_correlations = (
    regression_df[
        predictors
        +
        [target]
    ]
    .corr()[
        target
    ]
    .drop(
        target
    )
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)


print(
    "\n" + "=" * 72
)

print(
    "CORRELATION OF EACH PREDICTOR "
    "WITH GOALS SCORED"
)

print(
    "=" * 72
)


print(
    target_correlations
    .round(3)
    .to_string()
)


# ============================================================
# 14. TARGET SUMMARY
# ============================================================

print(
    "\nGoals-scored summary:"
)


print(
    regression_df[
        target
    ]
    .describe()
)


# ============================================================
# 15. PREDICTOR SUMMARY STATISTICS
# ============================================================

print(
    "\nPredictor summary statistics:"
)


print(
    regression_df[
        predictors
    ]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# 16. FIRST TOURNAMENT MATCH CHECK
# ============================================================
#
# This is useful because first-match rolling features
# should have been filled from pre-tournament baselines
# in the feature-building stage.
# ============================================================

first_match_rows = (
    df[
        df[
            "first_tournament_match"
        ]
        == 1
    ]
    if "first_tournament_match" in df.columns
    else pd.DataFrame()
)


if not first_match_rows.empty:

    print(
        "\nFirst-tournament-match observations:",
        len(first_match_rows)
    )


    first_match_missing = (
        first_match_rows[
            predictors
        ]
        .isna()
        .sum()
    )


    print(
        "\nMissing candidate predictors "
        "for first tournament matches:"
    )


    print(
        first_match_missing
    )


# ============================================================
# 17. SAVE DATASET
# ============================================================

regression_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n" + "=" * 72
)

print(
    "REGRESSION 2.2 DATASET SAVED"
)

print(
    "=" * 72
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nFinal candidate dataset:"
)

print(
    " - Rows:",
    len(regression_df)
)

print(
    " - Unique matches:",
    regression_df[
        "match_id"
    ].nunique()
)

print(
    " - Unique teams:",
    regression_df[
        "team"
    ].nunique()
)

print(
    " - Explanatory variables:",
    len(predictors)
)

print(
    " - Target:",
    target
)


print(
    "\nIMPORTANT:"
)

print(
    "These 8 predictors are still candidate variables."
)

print(
    "Review correlation and VIF before "
    "locking Regression 2.2."
)