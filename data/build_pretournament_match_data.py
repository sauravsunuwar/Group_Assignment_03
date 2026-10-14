import os

import pandas as pd





# ============================================================

# HIT140 ASSESSMENT 3

# PRE-TOURNAMENT MATCH DATA BUILDER

#

# Purpose:

# For each of the 48 FIFA World Cup 2026 teams,

# select the LAST 5 completed senior international matches

# played on calendar dates strictly BEFORE that team's

# first FIFA World Cup 2026 match.

#

# IMPORTANT:

# Historical results contain date-level information rather

# than reliable kickoff timestamps. Therefore, a historical

# match on the SAME CALENDAR DATE as the team's first World

# Cup match is excluded to prevent data leakage.

#

# Output:

# pre_tournament_matches.csv

# ============================================================





# ============================================================

# 1. FILE PATHS

# ============================================================



BASE_DIR = os.path.dirname(

    os.path.abspath(__file__)

)



DATA_DIR = os.path.join(

    BASE_DIR,

    "data"

)



MASTER_FILE = os.path.join(

    DATA_DIR,

    "world_cup_2026_master_matches.csv"

)



HISTORICAL_FILE = os.path.join(

    DATA_DIR,

    "international_results.csv"

)



OUTPUT_FILE = os.path.join(

    DATA_DIR,

    "pre_tournament_matches.csv"

)





# ============================================================

# 2. SETTINGS

# ============================================================



LAST_N_MATCHES = 5

# Global tournament cutoff: all historical matches must predate 11 June 2026.
GLOBAL_CUTOFF_DATE = pd.Timestamp('2026-06-11', tz='UTC').date()





# ============================================================

# 3. LOAD WORLD CUP MASTER DATABASE

# ============================================================



print("=" * 76)

print("HIT140 - PRE-TOURNAMENT MATCH DATA BUILDER")

print("=" * 76)



print(

    "\nLoading World Cup master database..."

)



master_df = pd.read_csv(

    MASTER_FILE

)



print(

    "World Cup matches:",

    len(master_df)

)





if len(master_df) != 104:



    raise ValueError(

        f"Expected 104 World Cup matches, "

        f"but found {len(master_df)}."

    )





# ============================================================

# 4. CONVERT WORLD CUP DATES

# ============================================================



master_df[

    "match_date"

] = pd.to_datetime(

    master_df[

        "match_date"

    ],

    errors="coerce",

    utc=True

)





if master_df[

    "match_date"

].isna().any():



    raise ValueError(

        "Some World Cup match dates "

        "could not be converted."

    )





print(

    "World Cup date conversion: PASSED"

)





# ============================================================

# 5. CREATE LIST OF 48 WORLD CUP TEAMS

# ============================================================



home_teams = set(

    master_df[

        "home_team"

    ].dropna()

)



away_teams = set(

    master_df[

        "away_team"

    ].dropna()

)



world_cup_teams = sorted(

    home_teams.union(

        away_teams

    )

)





print(

    "World Cup teams:",

    len(world_cup_teams)

)





if len(world_cup_teams) != 48:



    raise ValueError(

        f"Expected 48 World Cup teams, "

        f"but found {len(world_cup_teams)}."

    )





# ============================================================

# 6. FIND EACH TEAM'S FIRST WORLD CUP MATCH

# ============================================================



first_match_dates = {}





for team in world_cup_teams:



    team_matches = master_df[

        (

            master_df[

                "home_team"

            ]

            == team

        )

        |

        (

            master_df[

                "away_team"

            ]

            == team

        )

    ].copy()





    first_match_date = (

        team_matches[

            "match_date"

        ].min()

    )





    first_match_dates[

        team

    ] = first_match_date





print(

    "First World Cup match dates calculated:",

    len(first_match_dates)

)





# ============================================================

# 7. LOAD HISTORICAL INTERNATIONAL RESULTS

# ============================================================



print(

    "\nLoading historical international results..."

)





if not os.path.exists(

    HISTORICAL_FILE

):



    raise FileNotFoundError(

        "\nHistorical results file not found:\n"

        f"{HISTORICAL_FILE}\n\n"

        "Create/download international_results.csv "

        "before running this script."

    )





results_df = pd.read_csv(

    HISTORICAL_FILE

)





print(

    "Historical result rows:",

    len(results_df)

)





# ============================================================

# 8. VALIDATE HISTORICAL FILE

# ============================================================



required_columns = [

    "date",

    "home_team",

    "away_team",

    "home_score",

    "away_score"

]





missing_columns = [

    column



    for column in required_columns



    if column

    not in results_df.columns

]





if missing_columns:



    raise ValueError(

        "Historical results file is missing "

        f"columns: {missing_columns}"

    )





print(

    "Historical result columns: PASSED"

)





# ============================================================

# 9. PREPARE HISTORICAL DATES

# ============================================================



results_df[

    "date"

] = pd.to_datetime(

    results_df[

        "date"

    ],

    errors="coerce",

    utc=True

)





invalid_dates = (

    results_df[

        "date"

    ]

    .isna()

    .sum()

)





print(

    "Invalid historical dates:",

    invalid_dates

)





results_df = (

    results_df

    .dropna(

        subset=[

            "date"

        ]

    )

    .copy()

)





# ============================================================

# 10. ENSURE SCORES ARE NUMERIC

# ============================================================



results_df[

    "home_score"

] = pd.to_numeric(

    results_df[

        "home_score"

    ],

    errors="coerce"

)





results_df[

    "away_score"

] = pd.to_numeric(

    results_df[

        "away_score"

    ],

    errors="coerce"

)





results_df = (

    results_df

    .dropna(

        subset=[

            "home_score",

            "away_score"

        ]

    )

    .copy()

)





# ============================================================

# 11. TEAM NAME ALIASES

# ============================================================

#

# FIFA and the historical-results dataset use different names

# for a small number of national teams.

# ============================================================



TEAM_ALIASES = {



    "IR Iran":

        "Iran",



    "Korea Republic":

        "South Korea",



    "USA":

        "United States",



    "Cabo Verde":

        "Cape Verde",



    "Congo DR":

        "DR Congo",



    "Czechia":

        "Czech Republic",



    "Côte d'Ivoire":

        "Ivory Coast",



    "Türkiye":

        "Turkey"

}





# ============================================================

# 12. COLLECT LAST 5 PRE-WORLD-CUP MATCHES

# ============================================================



print(

    "\nCollecting last "

    f"{LAST_N_MATCHES} valid pre-World-Cup matches..."

)





# Remove ALL tournament-date records before selecting each team's matches.
excluded_count = (results_df['date'].dt.date >= GLOBAL_CUTOFF_DATE).sum()
results_df = results_df[results_df['date'].dt.date < GLOBAL_CUTOFF_DATE].copy()
print('Excluded on/after 11 June 2026:', excluded_count)

team_match_rows = []



missing_teams = []



insufficient_teams = []





for fifa_team in world_cup_teams:



    historical_team = (

        TEAM_ALIASES.get(

            fifa_team,

            fifa_team

        )

    )





    cutoff_timestamp = (

        first_match_dates[

            fifa_team

        ]

    )





    # --------------------------------------------------------

    # CRITICAL LEAKAGE-PREVENTION RULE

    # --------------------------------------------------------

    #

    # Historical data provides dates without reliable kickoff

    # times.

    #

    # Therefore:

    #

    # historical calendar date

    # MUST BE STRICTLY EARLIER THAN

    # first World Cup calendar date.

    #

    # Same-day historical records are excluded completely.

    # --------------------------------------------------------



    cutoff_calendar_date = GLOBAL_CUTOFF_DATE





    # --------------------------------------------------------

    # Find all historical matches involving this team

    # --------------------------------------------------------



    team_results = results_df[

        (

            results_df[

                "home_team"

            ]

            == historical_team

        )

        |

        (

            results_df[

                "away_team"

            ]

            == historical_team

        )

    ].copy()





    # --------------------------------------------------------

    # Keep ONLY matches on earlier calendar dates

    # --------------------------------------------------------



    team_results = team_results[

        team_results[

            "date"

        ].dt.date

        <

        cutoff_calendar_date

    ].copy()





    # --------------------------------------------------------

    # Sort newest first

    # --------------------------------------------------------



    team_results = (

        team_results

        .sort_values(

            "date",

            ascending=False

        )

    )





    # --------------------------------------------------------

    # Select last N valid historical matches

    # --------------------------------------------------------



    selected = (

        team_results

        .head(

            LAST_N_MATCHES

        )

        .copy()

    )





    if len(selected) == 0:



        missing_teams.append(

            fifa_team

        )



        continue





    if len(selected) < LAST_N_MATCHES:



        insufficient_teams.append(

            (

                fifa_team,

                len(selected)

            )

        )





    # --------------------------------------------------------

    # EXTRA SAFETY CHECK

    # --------------------------------------------------------



    invalid_selected = selected[

        selected[

            "date"

        ].dt.date

        >=

        cutoff_calendar_date

    ]





    if not invalid_selected.empty:



        raise ValueError(

            f"LEAKAGE DETECTED for {fifa_team}. "

            "A selected historical match is not "

            "strictly earlier than the team's "

            "global World Cup start date."

        )





    # --------------------------------------------------------

    # Convert matches to TEAM perspective

    # --------------------------------------------------------



    for _, match in selected.iterrows():



        if (

            match[

                "home_team"

            ]

            == historical_team

        ):



            opponent = (

                match[

                    "away_team"

                ]

            )



            goals_scored = int(

                match[

                    "home_score"

                ]

            )



            goals_conceded = int(

                match[

                    "away_score"

                ]

            )





        else:



            opponent = (

                match[

                    "home_team"

                ]

            )



            goals_scored = int(

                match[

                    "away_score"

                ]

            )



            goals_conceded = int(

                match[

                    "home_score"

                ]

            )





        team_match_rows.append(

            {



                "match_date":

                    match[

                        "date"

                    ].date(),



                "team":

                    fifa_team,



                "opponent":

                    opponent,



                "goals_scored":

                    goals_scored,



                "goals_conceded":

                    goals_conceded

            }

        )





# ============================================================

# 13. CREATE OUTPUT DATAFRAME

# ============================================================



output_df = pd.DataFrame(

    team_match_rows

)





# ============================================================

# 14. BASIC VALIDATION

# ============================================================



print(

    "\n" + "=" * 76

)



print(

    "VALIDATION"

)



print(

    "=" * 76

)





print(

    "\nWorld Cup teams:",

    len(world_cup_teams)

)





teams_collected = (

    output_df[

        "team"

    ].nunique()



    if not output_df.empty



    else 0

)





print(

    "Teams successfully collected:",

    teams_collected

)





expected_rows = (

    len(world_cup_teams)

    *

    LAST_N_MATCHES

)





print(

    "Expected team-match rows:",

    expected_rows

)





print(

    "Actual team-match rows:",

    len(output_df)

)





# ============================================================

# 15. MATCHES PER TEAM

# ============================================================



if not output_df.empty:



    match_counts = (

        output_df

        .groupby(

            "team"

        )

        .size()

    )





    print(

        "\nMatches per team:"

    )





    print(

        match_counts.describe()

    )





# ============================================================

# 16. MISSING / INSUFFICIENT TEAMS

# ============================================================



print(

    "\nTeams with ZERO historical matches:",

    len(missing_teams)

)





for team in missing_teams:



    print(

        " -",

        team

    )





print(

    "\nTeams with fewer than "

    f"{LAST_N_MATCHES} matches:",

    len(insufficient_teams)

)





for team, count in insufficient_teams:



    print(

        f" - {team}: {count}"

    )





# ============================================================

# 17. STRICT VALIDATION

# ============================================================



if missing_teams:



    raise ValueError(

        "Some World Cup teams have no historical "

        "matches. Check team-name aliases."

    )





if insufficient_teams:



    raise ValueError(

        "Some World Cup teams have fewer than "

        f"{LAST_N_MATCHES} valid historical matches."

    )





if len(output_df) != expected_rows:



    raise ValueError(

        f"Expected {expected_rows} rows, "

        f"but found {len(output_df)}."

    )





if teams_collected != 48:



    raise ValueError(

        f"Expected 48 teams, "

        f"but found {teams_collected}."

    )





# ============================================================

# 18. DUPLICATE CHECK

# ============================================================



duplicate_rows = (

    output_df

    .duplicated(

        subset=[

            "match_date",

            "team",

            "opponent"

        ]

    )

    .sum()

)





print(

    "\nDuplicate team-match rows:",

    duplicate_rows

)





if duplicate_rows != 0:



    raise ValueError(

        "Duplicate team-match observations detected."

    )





# ============================================================

# 19. FINAL CHRONOLOGICAL LEAKAGE CHECK

# ============================================================



output_df[

    "match_date"

] = pd.to_datetime(

    output_df[

        "match_date"

    ],

    errors="coerce"

)





leakage_rows = []





for team in world_cup_teams:



    team_output = output_df[

        output_df[

            "team"

        ]

        == team

    ].copy()





    first_wc_date = GLOBAL_CUTOFF_DATE





    invalid = team_output[

        team_output[

            "match_date"

        ].dt.date

        >=

        first_wc_date

    ]





    if not invalid.empty:



        invalid = invalid.copy()



        invalid[

            "global_cutoff_date"

        ] = first_wc_date



        leakage_rows.append(

            invalid

        )





if leakage_rows:



    leakage_df = pd.concat(

        leakage_rows,

        ignore_index=True

    )





    print(

        "\nLEAKAGE ROWS:"

    )





    print(

        leakage_df.to_string(

            index=False

        )

    )





    raise ValueError(

        "FINAL LEAKAGE CHECK FAILED."

    )





print(

    "\nChronological leakage check: PASSED"

)





# ============================================================

# 20. CHECK LATEST BASELINE DATE PER TEAM

# ============================================================



latest_dates = (

    output_df

    .groupby(

        "team"

    )[

        "match_date"

    ]

    .max()

)





same_or_later_count = 0





print(

    "\nLatest baseline match date checks:"

)





for team in world_cup_teams:



    latest_date = (

        latest_dates[

            team

        ].date()

    )





    first_wc_date = GLOBAL_CUTOFF_DATE





    if latest_date >= first_wc_date:



        same_or_later_count += 1



        print(

            f"WARNING: {team}: "

            f"{latest_date} >= {first_wc_date}"

        )





print(

    "Teams with latest baseline match "

    "on/after global World Cup start date:",

    same_or_later_count

)





if same_or_later_count != 0:



    raise ValueError(

        "Same-day or future baseline "

        "matches remain."

    )





# ============================================================

# 21. SORT OUTPUT

# ============================================================



output_df = (

    output_df

    .sort_values(

        [

            "team",

            "match_date"

        ]

    )

    .reset_index(

        drop=True

    )

)





# ============================================================

# 22. DISPLAY SAMPLE

# ============================================================



print(

    "\nExample observations:"

)





print(

    output_df

    .head(15)

    .to_string(

        index=False

    )

)





# ============================================================

# 23. SAVE

# ============================================================



output_df.to_csv(

    OUTPUT_FILE,

    index=False

)





print(

    "\n" + "=" * 76

)



print(

    "PRE-TOURNAMENT MATCH DATA SAVED"

)



print(

    "=" * 76

)





print(

    f"\nSaved to:\n{OUTPUT_FILE}"

)





print(

    "\nFinal rows:",

    len(output_df)

)





print(

    "Final teams:",

    output_df[

        "team"

    ].nunique()

)





print(

    "Matches per team:",

    LAST_N_MATCHES

)





print(

    "Same-day/future baseline matches:",

    same_or_later_count

)





print(

    "\nVALIDATION: PASSED"

)