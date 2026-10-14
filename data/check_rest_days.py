import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

# Folder containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Your CSV files are inside the nested "data" folder
DATA_DIR = BASE_DIR / "data"

master_path = DATA_DIR / "world_cup_2026_master_matches.csv"
history_path = DATA_DIR / "international_results.csv"

print("Master file:", master_path)
print("History file:", history_path)

# ============================================================
# LOAD DATA
# ============================================================

master = pd.read_csv(master_path)
history = pd.read_csv(history_path)

# Convert both date columns to the same timezone-naive format
master["match_date"] = pd.to_datetime(
master["match_date"],
    utc=True
).dt.tz_localize(None)

history["date"] = pd.to_datetime(
    history["date"],
    utc=True
).dt.tz_localize(None)

print("\nWorld Cup matches:", len(master))
print("Historical matches:", len(history))

# ============================================================
# TEAM NAME ALIASES
# ============================================================

# Some datasets use different names for the same national team.
# These aliases allow us to match the World Cup dataset with
# the historical international-results dataset.

aliases = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
}


def normalise_team(name):
    return aliases.get(name, name)


master["home_lookup"] = master["home_team"].apply(normalise_team)
master["away_lookup"] = master["away_team"].apply(normalise_team)

# ============================================================
# GET ALL WORLD CUP TEAMS
# ============================================================

teams = sorted(
    set(master["home_lookup"]).union(
        set(master["away_lookup"])
    )
)

print("World Cup teams found:", len(teams))

# ============================================================
# CHECK REST DAYS BEFORE EACH TEAM'S FIRST WORLD CUP MATCH
# ============================================================

first_matches = []

for team in teams:

    # Get all World Cup matches involving this team
    team_wc = master[
        (master["home_lookup"] == team)
        | (master["away_lookup"] == team)
    ].sort_values("match_date")

    # First World Cup match
    first_wc_date = team_wc.iloc[0]["match_date"]

    # Find historical matches involving this team
    # that occurred BEFORE the first World Cup match
    previous = history[
        (
            (history["home_team"] == team)
            | (history["away_team"] == team)
        )
        & (history["date"] < first_wc_date)
    ].sort_values("date")

    # --------------------------------------------------------
    # Calculate rest days
    # --------------------------------------------------------

    if len(previous) > 0:

        previous_date = previous.iloc[-1]["date"]

        rest_days = (
            first_wc_date - previous_date
        ).days

        status = "OK"

    else:

        previous_date = pd.NaT
        rest_days = None
        status = "MISSING"

    first_matches.append(
        {
            "team": team,
            "first_wc_match": first_wc_date.date(),
            "previous_match": (
                previous_date.date()
                if pd.notna(previous_date)
                else None
            ),
            "rest_days": rest_days,
            "status": status,
        }
    )

# ============================================================
# CREATE RESULTS TABLE
# ============================================================

result = pd.DataFrame(first_matches)

print("\n")
print("=" * 75)
print("FIRST WORLD CUP MATCH REST-DAY CHECK")
print("=" * 75)

print(result.to_string(index=False))

# ============================================================
# VALIDATION
# ============================================================

valid_count = (result["status"] == "OK").sum()
missing_count = (result["status"] == "MISSING").sum()

print("\n")
print("=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

print("Teams checked:", len(result))
print("Valid previous match:", valid_count)
print("Missing previous match:", missing_count)

# Check suspicious values
if valid_count > 0:

    valid_rest = result.loc[
        result["status"] == "OK",
        "rest_days"
    ]

    print("\nMinimum rest days:", valid_rest.min())
    print("Maximum rest days:", valid_rest.max())
    print("Average rest days:", round(valid_rest.mean(), 2))

    suspicious = result[
        (result["status"] == "OK")
        & (
            (result["rest_days"] <= 0)
            | (result["rest_days"] > 60)
        )
    ]

    print("\nSuspicious rest-day observations:", len(suspicious))

    if len(suspicious) > 0:
        print("\nSuspicious teams:")
        print(suspicious.to_string(index=False))

# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 75)

if missing_count == 0:

    print("PASS: Every World Cup team has a previous match available.")

else:

    print(
        "FAIL:",
        missing_count,
        "team(s) do not have a previous historical match."
    )

print("=" * 75)