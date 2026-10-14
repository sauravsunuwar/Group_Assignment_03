
import os
import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ============================================================
# HIT140 ASSESSMENT 3
# REGRESSION 2.1 - MULTICOLLINEARITY / VIF CHECK
# ============================================================


# ============================================================
# 1. LOCATE DATASET
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
        f"\nCould not find {filename}\n"
        f"Checked:\n"
        + "\n".join(possible_paths)
    )


DATA_FILE = find_file(
    "regression_2_1_dataset.csv"
)


print("=" * 70)
print("HIT140 - REGRESSION 2.1 VIF CHECK")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nRows loaded:", len(df))


if len(df) != 104:

    raise ValueError(
        f"Expected 104 matches, but found {len(df)}."
    )


if df["match_id"].nunique() != 104:

    raise ValueError(
        "Expected exactly 104 unique match IDs."
    )


print("Row-count validation: PASSED")


# ============================================================
# 3. DEFINE EXACTLY 8 PREDICTORS
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


print("\nPredictors:")

for i, predictor in enumerate(predictors, start=1):

    print(f"{i}. {predictor}")


if len(predictors) != 8:

    raise ValueError(
        "Exactly 8 predictors are required."
    )


# ============================================================
# 4. VALIDATE COLUMNS
# ============================================================

missing_columns = [
    predictor
    for predictor in predictors
    if predictor not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing predictor columns: {missing_columns}"
    )


print("\nPredictor-column validation: PASSED")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

missing_values = df[predictors].isna().sum()


print("\nMissing values:")
print(missing_values)


if missing_values.sum() != 0:

    raise ValueError(
        "Missing values detected. "
        "VIF calculation stopped."
    )


print("\nMissing-value validation: PASSED")


# ============================================================
# 6. CHECK CONSTANT VARIABLES
# ============================================================

unique_counts = df[predictors].nunique()


constant_variables = (
    unique_counts[unique_counts <= 1]
    .index
    .tolist()
)


if constant_variables:

    raise ValueError(
        f"Constant predictors detected: "
        f"{constant_variables}"
    )


print("\nConstant-variable check: PASSED")


# ============================================================
# 7. PREPARE NUMERIC MATRIX
# ============================================================

X = df[predictors].astype(float).copy()


if not np.isfinite(X.to_numpy()).all():

    raise ValueError(
        "Infinite or invalid numeric values detected."
    )


# ============================================================
# 8. CALCULATE VIF WITH INTERCEPT
# ============================================================

# Add a constant to calculate standard centered VIF.
# The constant itself is excluded from the reported VIFs.

X_with_constant = add_constant(
    X,
    has_constant="add"
)


vif_results = pd.DataFrame({

    "Predictor": predictors,

    "VIF": [

        variance_inflation_factor(
            X_with_constant.values,
            i + 1
        )

        for i in range(len(predictors))

    ]

})


# ============================================================
# 9. SORT RESULTS
# ============================================================

vif_results = (
    vif_results
    .sort_values(
        by="VIF",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("VIF RESULTS")
print("=" * 70)


print(
    vif_results.to_string(
        index=False,
        formatters={
            "VIF": "{:.3f}".format
        }
    )
)


# ============================================================
# 10. VIF DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VIF DIAGNOSTIC SUMMARY")
print("=" * 70)


low_vif = vif_results[
    vif_results["VIF"] < 5
]


moderate_vif = vif_results[
    (vif_results["VIF"] >= 5)
    &
    (vif_results["VIF"] < 10)
]


high_vif = vif_results[
    vif_results["VIF"] >= 10
]


print("\nVIF < 5:")

if not low_vif.empty:

    for _, row in low_vif.iterrows():

        print(
            f" - {row['Predictor']}: "
            f"{row['VIF']:.3f}"
        )

else:

    print("None")


print("\nVIF 5 to <10:")

if not moderate_vif.empty:

    for _, row in moderate_vif.iterrows():

        print(
            f" - {row['Predictor']}: "
            f"{row['VIF']:.3f}"
        )

else:

    print("None")


print("\nVIF >= 10:")

if not high_vif.empty:

    for _, row in high_vif.iterrows():

        print(
            f" - {row['Predictor']}: "
            f"{row['VIF']:.3f}"
        )

else:

    print("None")


# ============================================================
# 11. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)


correlation_matrix = X.corr()


print(
    correlation_matrix
    .round(3)
    .to_string()
)


# ============================================================
# 12. HIGH CORRELATION PAIRS
# ============================================================

print(
    "\nPredictor pairs with "
    "|correlation| >= 0.80:"
)


high_corr_pairs = []


for i in range(len(predictors)):

    for j in range(i + 1, len(predictors)):

        predictor_a = predictors[i]
        predictor_b = predictors[j]

        correlation = correlation_matrix.loc[
            predictor_a,
            predictor_b
        ]

        if abs(correlation) >= 0.80:

            high_corr_pairs.append(
                (
                    predictor_a,
                    predictor_b,
                    correlation
                )
            )


if high_corr_pairs:

    for predictor_a, predictor_b, correlation in high_corr_pairs:

        print(
            f" - {predictor_a} "
            f"<-> {predictor_b}: "
            f"{correlation:.3f}"
        )

else:

    print("None")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VIF CHECK COMPLETE")
print("=" * 70)


print("\nRows checked:", len(df))
print("Predictors checked:", len(predictors))

print(
    "Highest VIF:",
    round(vif_results["VIF"].max(), 3)
)


print("\nIMPORTANT:")

print(
    "Do not automatically remove a predictor "
    "based only on VIF."
)

print(
    "Interpret VIF together with correlations, "
    "football rationale, pre-match validity "
    "and the Assessment 3 requirements."
)
