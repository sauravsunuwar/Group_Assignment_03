import os
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# HIT140 ASSESSMENT 3
# REGRESSION 2.2 - MULTICOLLINEARITY / VIF CHECK
# ============================================================
#
# Target:
# goals_scored
#
# Dataset:
# 208 team-match observations
#
# Purpose:
# Check whether the 8 candidate explanatory variables
# have problematic multicollinearity.
# ============================================================


# ============================================================
# 1. LOCATE DATASET
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def find_file(filename):

    possible_paths = [
        os.path.join(
            BASE_DIR,
            filename
        ),

        os.path.join(
            BASE_DIR,
            "data",
            filename
        )
    ]

    for path in possible_paths:

        if os.path.exists(path):

            return path

    raise FileNotFoundError(
        f"\nCould not find: {filename}\n"
        f"Checked:\n"
        + "\n".join(possible_paths)
    )


DATA_FILE = find_file(
    "regression_2_2_dataset.csv"
)


# ============================================================
# 2. START
# ============================================================

print("=" * 72)

print(
    "HIT140 - REGRESSION 2.2 VIF CHECK"
)

print("=" * 72)


print(
    "\nDataset:"
)

print(
    DATA_FILE
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


print(
    "\nRows loaded:",
    len(df)
)


# Regression 2.2 must contain 208 team-match observations

if len(df) != 208:

    raise ValueError(
        f"Regression 2.2 should contain "
        f"208 rows, but found {len(df)}."
    )


print(
    "Row-count validation: PASSED"
)


# ============================================================
# 4. DEFINE EXACTLY 8 PREDICTORS
# ============================================================

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


print(
    "\nCandidate predictors:"
)


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
# 5. VALIDATE COLUMNS
# ============================================================

missing_columns = [

    predictor

    for predictor in predictors

    if predictor not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing predictor columns: "
        f"{missing_columns}"
    )


print(
    "\nPredictor-column validation: PASSED"
)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

missing_values = (
    df[
        predictors
    ]
    .isna()
    .sum()
)


print(
    "\nMissing values:"
)


print(
    missing_values
)


if missing_values.sum() != 0:

    raise ValueError(
        "Missing predictor values detected. "
        "VIF calculation stopped."
    )


print(
    "\nMissing-value validation: PASSED"
)


# ============================================================
# 7. CHECK CONSTANT VARIABLES
# ============================================================

unique_counts = (
    df[
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


constant_variables = (
    unique_counts[
        unique_counts <= 1
    ]
    .index
    .tolist()
)


if constant_variables:

    raise ValueError(
        f"Constant predictors detected: "
        f"{constant_variables}"
    )


print(
    "\nConstant-variable check: PASSED"
)


# ============================================================
# 8. PREPARE NUMERIC MATRIX
# ============================================================

X = (
    df[
        predictors
    ]
    .astype(float)
    .copy()
)


# ============================================================
# 9. CALCULATE VIF
# ============================================================

vif_results = pd.DataFrame()


vif_results[
    "Predictor"
] = predictors


vif_results[
    "VIF"
] = [

    variance_inflation_factor(
        X.values,
        i
    )

    for i in range(
        X.shape[1]
    )
]


# ============================================================
# 10. SORT VIF RESULTS
# ============================================================

vif_results = (
    vif_results
    .sort_values(
        by="VIF",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n" + "=" * 72
)

print(
    "VIF RESULTS"
)

print(
    "=" * 72
)


print(
    vif_results.to_string(
        index=False,
        formatters={
            "VIF":
                "{:.3f}".format
        }
    )
)


# ============================================================
# 11. VIF DIAGNOSTIC GROUPS
# ============================================================

low_vif = (
    vif_results[
        vif_results[
            "VIF"
        ] < 5
    ]
)


moderate_vif = (
    vif_results[
        (
            vif_results[
                "VIF"
            ] >= 5
        )
        &
        (
            vif_results[
                "VIF"
            ] < 10
        )
    ]
)


high_vif = (
    vif_results[
        vif_results[
            "VIF"
        ] >= 10
    ]
)


print(
    "\n" + "=" * 72
)

print(
    "VIF DIAGNOSTIC SUMMARY"
)

print(
    "=" * 72
)


# ============================================================
# 12. VIF < 5
# ============================================================

print(
    "\nVIF < 5:"
)


if len(low_vif) > 0:

    for _, row in low_vif.iterrows():

        print(
            f" - {row['Predictor']}: "
            f"{row['VIF']:.3f}"
        )

else:

    print(
        "None"
    )


# ============================================================
# 13. VIF 5 TO < 10
# ============================================================

print(
    "\nVIF 5 to <10:"
)


if len(moderate_vif) > 0:

    for _, row in moderate_vif.iterrows():

        print(
            f" - {row['Predictor']}: "
            f"{row['VIF']:.3f}"
        )

else:

    print(
        "None"
    )


# ============================================================
# 14. VIF >= 10
# ============================================================

print(
    "\nVIF >= 10:"
)


if len(high_vif) > 0:

    for _, row in high_vif.iterrows():

        print(
            f" - {row['Predictor']}: "
            f"{row['VIF']:.3f}"
        )

else:

    print(
        "None"
    )


# ============================================================
# 15. CORRELATION MATRIX
# ============================================================

correlation_matrix = (
    X.corr()
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
# 16. HIGH-CORRELATION PAIRS
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

        predictor_a = (
            predictors[i]
        )

        predictor_b = (
            predictors[j]
        )


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
# 17. CORRELATION WITH TARGET
# ============================================================

target = "goals_scored"


if target not in df.columns:

    raise ValueError(
        f"Target column '{target}' "
        f"was not found."
    )


target_correlations = (
    df[
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
    "CORRELATION WITH GOALS SCORED"
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
# 18. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "VIF CHECK COMPLETE"
)

print(
    "=" * 72
)


print(
    "\nRows checked:",
    len(df)
)


print(
    "Predictors checked:",
    len(predictors)
)


print(
    "Highest VIF:",
    round(
        vif_results[
            "VIF"
        ].max(),
        3
    )
)


print(
    "\nIMPORTANT:"
)


print(
    "Do not automatically remove a predictor "
    "based only on VIF."
)


print(
    "Interpret VIF together with correlations, "
    "football rationale, pre-match validity "
    "and the Assessment 3 requirements."
)