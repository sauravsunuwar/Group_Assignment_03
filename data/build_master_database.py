import os
import time
import requests
import pandas as pd


# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

output_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(output_dir, "data")
os.makedirs(data_dir, exist_ok=True)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}


# --------------------------------------------------
# 2. FIFA WORLD CUP 2026 DETAILS
# --------------------------------------------------

COMPETITION_ID = 17
SEASON_ID = 285023

matches_url = (
    "https://api.fifa.com/api/v3/calendar/matches"
    f"?language=en&count=500"
    f"&IdCompetition={COMPETITION_ID}"
    f"&IdSeason={SEASON_ID}"
)


# --------------------------------------------------
# 3. GET TOURNAMENT MATCH LIST
# --------------------------------------------------

print("Downloading FIFA World Cup 2026 match list...")

response = requests.get(
    matches_url,
    headers=headers,
    timeout=30
)

response.raise_for_status()

matches_data = response.json()

matches = matches_data.get("Results", [])

print("Total matches found:", len(matches))


# --------------------------------------------------
# 4. FUNCTION TO GET FULL MATCH INFORMATION
# --------------------------------------------------

def get_match_details(match):

    match_id = match["IdMatch"]
    stage_id = match["IdStage"]

    # Get stage name
    stage_info = match.get("StageName", [])

    if stage_info:
        stage = stage_info[0].get("Description", "Unknown")
    else:
        stage = "Unknown"

    # FIFA live match endpoint
    match_url = (
        "https://api.fifa.com/api/v3/live/football/"
        f"{COMPETITION_ID}/{SEASON_ID}/"
        f"{stage_id}/{match_id}"
        "?language=en"
    )

    response = requests.get(
        match_url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    match_data = response.json()

    # ----------------------------------------------
    # Team information
    # ----------------------------------------------

    home_team = match_data.get("HomeTeam", {})
    away_team = match_data.get("AwayTeam", {})

    home_team_id = home_team.get("IdTeam")
    away_team_id = away_team.get("IdTeam")

    home_name_data = home_team.get("TeamName", [])
    away_name_data = away_team.get("TeamName", [])

    if home_name_data:
        home_team_name = home_name_data[0].get(
            "Description",
            str(home_team_id)
        )
    else:
        home_team_name = str(home_team_id)

    if away_name_data:
        away_team_name = away_name_data[0].get(
            "Description",
            str(away_team_id)
        )
    else:
        away_team_name = str(away_team_id)

    # ----------------------------------------------
    # Goals
    # ----------------------------------------------

    home_goals = home_team.get("Score")
    away_goals = away_team.get("Score")

    # ----------------------------------------------
    # Match date
    # ----------------------------------------------

    match_date = match_data.get("Date")

    if match_date is None:
        match_date = match.get("Date")

    # ----------------------------------------------
    # Goal difference
    #
    # Positive = home team scored more
    # Negative = away team scored more
    # Zero = draw
    # ----------------------------------------------

    if home_goals is not None and away_goals is not None:
        goal_difference = home_goals - away_goals
    else:
        goal_difference = None

    # ----------------------------------------------
    # Return one row per match
    # ----------------------------------------------

    return {
        "match_id": match_id,
        "match_date": match_date,
        "stage": stage,
        "home_team_id": home_team_id,
        "home_team": home_team_name,
        "away_team_id": away_team_id,
        "away_team": away_team_name,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "goal_difference": goal_difference
    }


# --------------------------------------------------
# 5. COLLECT ALL 104 MATCHES
# --------------------------------------------------

all_matches = []
failed_matches = []

for i, match in enumerate(matches, start=1):

    match_id = match.get("IdMatch")

    print(
        f"Collecting match {i} of {len(matches)} "
        f"(Match ID: {match_id})..."
    )

    try:

        row = get_match_details(match)
        all_matches.append(row)

    except requests.exceptions.RequestException as error:

        print("Request failed.")
        print("Retrying in 5 seconds...")

        time.sleep(5)

        try:

            row = get_match_details(match)
            all_matches.append(row)

        except requests.exceptions.RequestException as second_error:

            print(
                f"Could not collect match {match_id}"
            )

            print("Error:", second_error)

            failed_matches.append(match_id)

    # Small delay so we do not send requests too quickly
    time.sleep(0.5)


# --------------------------------------------------
# 6. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(all_matches)

print("\n--------------------------------")
print("MASTER DATABASE CREATED")
print("--------------------------------")

print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 7. CLEAN DATE COLUMN
# --------------------------------------------------

if "match_date" in df.columns:

    df["match_date"] = pd.to_datetime(
        df["match_date"],
        errors="coerce"
    )

    # Sort chronologically
    df = df.sort_values(
        by=["match_date", "match_id"]
    ).reset_index(drop=True)


# --------------------------------------------------
# 8. DATA VALIDATION
# --------------------------------------------------

print("\n--------------------------------")
print("DATA VALIDATION")
print("--------------------------------")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate match IDs:")
print(df["match_id"].duplicated().sum())

print("\nNumber of unique matches:")
print(df["match_id"].nunique())

print("\nStage counts:")
print(df["stage"].value_counts())

print("\nGoal difference summary:")
print(df["goal_difference"].describe())


# --------------------------------------------------
# 9. CHECK EXPECTED NUMBER OF MATCHES
# --------------------------------------------------

if len(df) == 104:

    print(
        "\nSUCCESS: Dataset contains exactly "
        "104 FIFA World Cup 2026 matches."
    )

else:

    print(
        "\nWARNING: Expected 104 matches, "
        f"but collected {len(df)}."
    )


# --------------------------------------------------
# 10. CHECK FAILED MATCHES
# --------------------------------------------------

print("\nFailed matches:", len(failed_matches))

if failed_matches:

    print("Failed Match IDs:")

    for match_id in failed_matches:
        print(match_id)


# --------------------------------------------------
# 11. DISPLAY FIRST 10 ROWS
# --------------------------------------------------

print("\n--------------------------------")
print("FIRST 10 MATCHES")
print("--------------------------------")

print(
    df[
        [
            "match_id",
            "match_date",
            "stage",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "goal_difference"
        ]
    ].head(10)
)


# --------------------------------------------------
# 12. SAVE MASTER DATABASE
# --------------------------------------------------

output_file = os.path.join(
    data_dir,
    "world_cup_2026_master_matches.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n--------------------------------")
print("FILE SAVED")
print("--------------------------------")

print(
    "Master database saved to:"
)

print(output_file)