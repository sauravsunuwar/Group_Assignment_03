import os
import re
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# HIT140 ASSESSMENT 3
# FIFA WORLD CUP 2026 - PRE-TOURNAMENT SQUAD MARKET VALUES
#
# Purpose:
# Build one squad-value observation for each of the 48 teams.
#
# Predictor later used in Regression 2.1:
#
# log_squad_value_difference
#     = log(home squad value) - log(away squad value)
#
# IMPORTANT:
# Market values must represent the pre-tournament period.
# They must NOT be values updated using World Cup performance.
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MASTER_FILE = os.path.join(
    DATA_DIR,
    "world_cup_2026_master_matches.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "world_cup_2026_squad_values.csv"
)


# ============================================================
# 2. TRANSFERMARKT SOURCE
# ============================================================

URL = (
    "https://www.transfermarkt.com/"
    "weltmeisterschaft/teilnehmer/pokalwettbewerb/FIWC"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# 3. TEAM NAME STANDARDISATION
# ============================================================

TEAM_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",

    "Korea Republic": "South Korea",
    "Korea, South": "South Korea",

    "IR Iran": "Iran",
    "Iran, Islamic Republic of": "Iran",

    "Turkey": "Türkiye",
    "Türkiye": "Türkiye",

    "Czech Republic": "Czechia",

    "Congo DR": "Congo DR",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",

    "Cape Verde": "Cabo Verde",

    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",

    "Curacao": "Curaçao",

    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def standardise_team_name(team):

    if pd.isna(team):
        return team

    import unicodedata

    # Convert to string and remove leading/trailing spaces
    team = str(team).strip()

    # Normalise Unicode characters
    team = unicodedata.normalize("NFC", team)

    # Remove non-breaking and zero-width spaces
    team = (
        team
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .strip()
    )

    # Collapse repeated spaces
    team = " ".join(team.split())

    aliases = {
        "USA": "United States",
        "United States of America": "United States",

        "Korea Republic": "South Korea",
        "Korea, South": "South Korea",

        "IR Iran": "Iran",
        "Iran, Islamic Republic of": "Iran",

        "Turkey": "Türkiye",
        "Turkiye": "Türkiye",
        "Türkiye": "Türkiye",

        "Czech Republic": "Czechia",

        "DR Congo": "Congo DR",
        "Democratic Republic of the Congo": "Congo DR",

        "Cape Verde": "Cabo Verde",

        "Ivory Coast": "Côte d'Ivoire",
        "Cote d'Ivoire": "Côte d'Ivoire",

        "Curacao": "Curaçao",

        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    }

    return aliases.get(team, team)

# ============================================================
# 4. MARKET VALUE CONVERTER
# ============================================================

def convert_market_value(value):
    """
    Convert Transfermarkt market value strings into EUR millions.

    Examples:
        €1.52bn  -> 1520.0
        €850.00m -> 850.0
        €25.50m  -> 25.5
        €750k    -> 0.75
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    value = (
        value
        .replace("€", "")
        .replace("£", "")
        .replace("$", "")
        .replace(",", "")
        .replace("\xa0", "")
        .strip()
        .lower()
    )

    # Remove spaces
    value = value.replace(" ", "")

    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(bn|b|m|k)?",
        value
    )

    if not match:
        return np.nan

    number = float(match.group(1))

    unit = match.group(2)

    if unit in ["bn", "b"]:
        return number * 1000

    elif unit == "m":
        return number

    elif unit == "k":
        return number / 1000

    else:
        return number


# ============================================================
# 5. LOAD MASTER WORLD CUP DATABASE
# ============================================================

print("=" * 72)
print("HIT140 - WORLD CUP 2026 SQUAD VALUE BUILDER")
print("=" * 72)

print("\nLoading World Cup master database...")

if not os.path.exists(MASTER_FILE):

    raise FileNotFoundError(
        f"Master database not found:\n{MASTER_FILE}"
    )


master = pd.read_csv(
    MASTER_FILE
)

print("World Cup matches:", len(master))


required_master_columns = [
    "home_team",
    "away_team"
]


missing_master_columns = [
    column
    for column in required_master_columns
    if column not in master.columns
]


if missing_master_columns:

    raise ValueError(
        f"Missing master columns: {missing_master_columns}"
    )


# ============================================================
# 6. GET THE 48 WORLD CUP TEAMS
# ============================================================

world_cup_teams = sorted(
    set(
        master["home_team"]
        .dropna()
        .apply(standardise_team_name)
    )
    |
    set(
        master["away_team"]
        .dropna()
        .apply(standardise_team_name)
    )
)


print(
    "Unique World Cup teams:",
    len(world_cup_teams)
)


if len(world_cup_teams) != 48:

    print(
        "\nWARNING:"
        f" Expected 48 teams but found {len(world_cup_teams)}."
    )


# ============================================================
# 7. DOWNLOAD TRANSFERMARKT PAGE
# ============================================================

print("\nDownloading Transfermarkt participant table...")


response = requests.get(
    URL,
    headers=HEADERS,
    timeout=30
)


print(
    "HTTP status:",
    response.status_code
)


response.raise_for_status()


# Small delay - polite request behaviour
time.sleep(1)


# ============================================================
# 8. FIRST TRY: READ HTML TABLES WITH PANDAS
# ============================================================

print(
    "\nSearching HTML tables..."
)


tables = []


try:

    tables = pd.read_html(
        response.text
    )

except Exception as error:

    print(
        "pd.read_html could not parse tables:",
        error
    )


print(
    "HTML tables found:",
    len(tables)
)


for i, table in enumerate(tables):

    print(
        f"\nTable {i}:",
        table.shape
    )

    print(
        "Columns:",
        list(table.columns)
    )


# ============================================================
# 9. EXTRACT TEAM + TOTAL MARKET VALUE
# ============================================================

records = []


for table in tables:

    temp = table.copy()

    # Convert possible MultiIndex headers to strings
    if isinstance(
        temp.columns,
        pd.MultiIndex
    ):

        temp.columns = [
            " ".join(
                [
                    str(part)
                    for part in column
                    if str(part) != "nan"
                ]
            ).strip()
            for column in temp.columns
        ]

    else:

        temp.columns = [
            str(column).strip()
            for column in temp.columns
        ]


    # --------------------------------------------------------
    # Find possible club/team column
    # --------------------------------------------------------

    team_column = None

    for column in temp.columns:

        column_lower = column.lower()

        if (
            "club" in column_lower
            or
            "team" in column_lower
            or
            "participant" in column_lower
        ):

            team_column = column
            break


    # --------------------------------------------------------
    # Find possible total market value column
    # --------------------------------------------------------

    value_column = None

    for column in temp.columns:

        column_lower = column.lower()

        if (
            "total market value" in column_lower
            or
            "market value" in column_lower
        ):

            value_column = column

            # Prefer total market value rather than average
            if (
                "total" in column_lower
            ):

                break


    if (
        team_column is None
        or
        value_column is None
    ):

        continue


    print(
        "\nPossible Transfermarkt table found."
    )

    print(
        "Team column:",
        team_column
    )

    print(
        "Market-value column:",
        value_column
    )


    for _, row in temp.iterrows():

        team = row[
            team_column
        ]

        value = row[
            value_column
        ]

        if pd.isna(team):
            continue

        team = standardise_team_name(
            team
        )

        market_value = convert_market_value(
            value
        )

        if pd.isna(market_value):
            continue

        records.append(
            {
                "team": team,
                "squad_value_eur_millions": market_value
            }
        )


# ============================================================
# 10. FALLBACK: PARSE TRANSFERMARKT HTML DIRECTLY
# ============================================================

if len(records) < 40:

    print(
        "\nTable extraction did not return enough teams."
    )

    print(
        "Trying direct HTML extraction..."
    )


    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )


    rows = soup.select(
        "table.items > tbody > tr"
    )


    print(
        "Transfermarkt rows found:",
        len(rows)
    )


    html_records = []


    for row in rows:

        # ----------------------------------------------------
        # Team name
        # ----------------------------------------------------

        team = None


        team_links = row.select(
            "td.hauptlink a"
        )


        for link in team_links:

            candidate = (
                link.get_text(
                    " ",
                    strip=True
                )
            )

            href = link.get(
                "href",
                ""
            )


            if (
                candidate
                and
                (
                    "/verein/" in href
                    or
                    "/startseite/verein/" in href
                )
            ):

                team = candidate
                break


        if team is None:
            continue


        team = standardise_team_name(
            team
        )


        # ----------------------------------------------------
        # Find monetary values in row
        # ----------------------------------------------------

        cells = row.find_all(
            "td"
        )


        money_values = []


        for cell in cells:

            text = cell.get_text(
                " ",
                strip=True
            )


            if "€" in text:

                converted = convert_market_value(
                    text
                )


                if not pd.isna(
                    converted
                ):

                    money_values.append(
                        converted
                    )


        if not money_values:
            continue


        # Participant tables normally contain average market
        # value and total market value.
        #
        # Total squad value should be the largest monetary
        # number in the row.
        total_value = max(
            money_values
        )


        html_records.append(
            {
                "team": team,
                "squad_value_eur_millions": total_value
            }
        )


    if len(html_records) > len(records):

        records = html_records


# ============================================================
# 11. CREATE DATAFRAME
# ============================================================

squad_df = pd.DataFrame(
    records
)


if squad_df.empty:

    raise ValueError(
        "\nNo squad market values could be extracted.\n"
        "Transfermarkt may have blocked the request or "
        "changed its page structure.\n\n"
        "DO NOT invent the missing values."
    )


# ============================================================
# 12. CLEAN EXTRACTED TEAM NAMES
# ============================================================

squad_df["team"] = (
    squad_df["team"]
    .apply(
        standardise_team_name
    )
)


# Remove duplicate team rows
squad_df = (
    squad_df
    .sort_values(
        "squad_value_eur_millions",
        ascending=False
    )
    .drop_duplicates(
        subset="team",
        keep="first"
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nUnique teams extracted:",
    squad_df["team"].nunique()
)


# ============================================================
# 13. MATCH ONLY WORLD CUP TEAMS
# ============================================================

world_cup_set = set(
    world_cup_teams
)


extracted_set = set(
    squad_df["team"]
)


missing_teams = sorted(
    world_cup_set
    -
    extracted_set
)


extra_teams = sorted(
    extracted_set
    -
    world_cup_set
)


print(
    "\nWorld Cup teams missing "
    "from extracted squad values:"
)


if missing_teams:

    for team in missing_teams:
        print(" -", team)

else:

    print("None")


print(
    "\nExtracted teams not present "
    "in World Cup master database:"
)


if extra_teams:

    for team in extra_teams:
        print(" -", team)

else:

    print("None")


# Keep only actual World Cup teams
squad_df = squad_df[
    squad_df["team"].isin(
        world_cup_set
    )
].copy()


# ============================================================
# 14. STRICT COVERAGE VALIDATION
# ============================================================

if len(missing_teams) > 0:

    raise ValueError(
        "\nSTOPPING: squad values are incomplete.\n"
        f"Missing {len(missing_teams)} World Cup team(s).\n"
        "Do NOT fill these with guessed values."
    )


if squad_df["team"].nunique() != 48:

    raise ValueError(
        "\nExpected exactly 48 World Cup teams, "
        f"but found {squad_df['team'].nunique()}."
    )


print(
    "\n48-team coverage validation: PASSED"
)


# ============================================================
# 15. VALIDATE MARKET VALUES
# ============================================================

if (
    squad_df[
        "squad_value_eur_millions"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "Missing squad market values detected."
    )


if (
    squad_df[
        "squad_value_eur_millions"
    ]
    <= 0
).any():

    raise ValueError(
        "Non-positive squad market values detected."
    )


print(
    "Market-value validation: PASSED"
)


# ============================================================
# 16. CREATE LOG SQUAD VALUE
# ============================================================
#
# Natural log reduces the extreme skew caused by very large
# differences between high-value and low-value squads.
#
# We use log1p:
#
# log(1 + squad value in EUR millions)
#
# ============================================================

squad_df[
    "log_squad_value"
] = np.log1p(
    squad_df[
        "squad_value_eur_millions"
    ]
)


# ============================================================
# 17. FINAL COLUMN SELECTION
# ============================================================

squad_df = squad_df[
    [
        "team",
        "squad_value_eur_millions",
        "log_squad_value"
    ]
].copy()


squad_df = (
    squad_df
    .sort_values(
        "squad_value_eur_millions",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 18. FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 72)

print(
    "SQUAD VALUE VALIDATION"
)

print("=" * 72)


print(
    "\nRows:",
    len(squad_df)
)


print(
    "Unique teams:",
    squad_df[
        "team"
    ].nunique()
)


print(
    "Duplicate teams:",
    squad_df[
        "team"
    ].duplicated().sum()
)


print(
    "\nMissing values:"
)


print(
    squad_df
    .isna()
    .sum()
)


print(
    "\nSquad value summary "
    "(EUR millions):"
)


print(
    squad_df[
        "squad_value_eur_millions"
    ]
    .describe()
)


print(
    "\nTop 10 squad values:"
)


print(
    squad_df
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nLowest 10 squad values:"
)


print(
    squad_df
    .tail(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE
# ============================================================

squad_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 72)

print(
    "SQUAD VALUE FILE SAVED"
)

print("=" * 72)


print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


print(
    "\nColumns created:"
)

print(
    " - team"
)

print(
    " - squad_value_eur_millions"
)

print(
    " - log_squad_value"
)


print(
    "\nIMPORTANT:"
)

print(
    "Before using this predictor in the final assessment, "
    "confirm that the extracted Transfermarkt values represent "
    "the pre-tournament valuation period."
)

print(
    "Do not use market values updated using World Cup "
    "performance."
)