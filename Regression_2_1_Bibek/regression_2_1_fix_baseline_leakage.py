"""Safely correct first-match baseline leakage in Regression 2.1's supplied CSV.

Creates NEW CSV without modifying the team's source data.
Evidence: several World Cup fixtures appeared in pre_tournament_matches.csv,
and their outcomes were included in the baseline statistics used to build the
predictors for those very matches.

Usage:
  python regression_2_1_fix_baseline_leakage.py \
      --regression regression_2_1_dataset.csv \
      --pre-matches pre_tournament_matches.csv \
      --baselines pre_tournament_baselines.csv \
      --master world_cup_2026_master_matches.csv \
      --output regression_2_1_dataset_corrected.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = "2026-06-11"  # Opening date. Raw pre-match file supplies date-only values.
NAME_ALIASES = {
    "South Korea": "Korea Republic",
    "Czech Republic": "Czechia",
    "United States": "USA",
    "Turkey": "Türkiye",
    "Iran": "IR Iran",
}
FIELDS = {
    "points_per_match_difference": "baseline_points_per_match",
    "avg_goals_scored_difference": "baseline_avg_goals_scored",
    "avg_goals_conceded_difference": "baseline_avg_goals_conceded",
    "scoring_match_rate_difference": "baseline_scoring_match_rate",
    "scoring_consistency_difference": "baseline_scoring_consistency",
    "clean_sheet_rate_difference": "baseline_clean_sheet_rate",
}


def recreate_team_baselines(prematches):
    rows = []
    for team, g in prematches.groupby("team", sort=True):
        goals_for = g["goals_scored"].to_numpy(dtype=float)
        goals_against = g["goals_conceded"].to_numpy(dtype=float)
        rows.append({
            "team": team,
            "baseline_matches": len(g),
            "baseline_avg_goals_scored": goals_for.mean(),
            "baseline_avg_goals_conceded": goals_against.mean(),
            "baseline_win_rate": (goals_for > goals_against).mean(),
            "baseline_points_per_match": (3 * (goals_for > goals_against) + (goals_for == goals_against)).mean(),
            "baseline_avg_goal_difference": (goals_for - goals_against).mean(),
            "baseline_scoring_match_rate": (goals_for > 0).mean(),
            "baseline_clean_sheet_rate": (goals_against == 0).mean(),
            "baseline_scoring_consistency": goals_for.std(ddof=0),
        })
    return pd.DataFrame(rows).set_index("team")


def correction(regression_path, pre_path, baseline_path, master_path, output_path):
    reg = pd.read_csv(regression_path)
    pre = pd.read_csv(pre_path)
    existing = pd.read_csv(baseline_path).set_index("team")
    master = pd.read_csv(master_path)
    if not pre.match_date.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        raise ValueError("Expected date-only pre_tournament_matches.csv dates")
    if not reg.match_id.is_unique:
        raise ValueError("Regression match_id must be unique")
    # Verify that the supplied team baselines actually came from ALL rows,
    # including the late tournament records, before attributing leakage.
    reconstructed = recreate_team_baselines(pre)
    if set(reconstructed.index) != set(existing.index):
        raise ValueError("Baseline teams do not match")
    for field in reconstructed.columns:
        if not np.allclose(reconstructed.loc[existing.index, field], existing[field], atol=1e-9):
            raise ValueError(f"Cannot reconstruct original baseline statistic {field}")

    late = pre[pre.match_date >= CUTOFF].copy()
    for col in ["team", "opponent"]:
        late[col + "_canonical"] = late[col].replace(NAME_ALIASES)
    matched_ids = []
    for _, item in late.iterrows():
        home = (master.home_team.eq(item.team_canonical)
                & master.away_team.eq(item.opponent_canonical)
                & master.home_goals.eq(item.goals_scored)
                & master.away_goals.eq(item.goals_conceded))
        away = (master.away_team.eq(item.team_canonical)
                & master.home_team.eq(item.opponent_canonical)
                & master.away_goals.eq(item.goals_scored)
                & master.home_goals.eq(item.goals_conceded))
        m = master[home | away]
        if len(m) != 1:
            raise ValueError(f"Late baseline row does not match one tournament fixture: {item.team} - {item.opponent}")
        matched_ids.append(int(m.iloc[0].match_id))

    ids = sorted(set(matched_ids))
    if len(matched_ids) != 2 * len(ids):
        raise ValueError("Expected two team-perspective rows per tournament fixture")
    safe = recreate_team_baselines(pre[pre.match_date < CUTOFF])
    if set(safe.index) != set(existing.index):
        raise ValueError("Insufficient pre-tournament observations for some teams")

    corrected = reg.copy()
    changed_features = 0
    for match_id in ids:
        mask = reg.match_id == match_id
        if mask.sum() != 1:
            raise ValueError(f"Affected match not unique in regression dataset: {match_id}")
        original = reg.loc[mask].iloc[0]
        ht, at = original.home_team, original.away_team
        for field, base in FIELDS.items():
            expected_original = existing.loc[ht, base] - existing.loc[at, base]
            if not np.isclose(original[field], expected_original, atol=1e-9):
                raise ValueError(f"Unexpected original input for match {match_id}, field {field}")
            adjusted = safe.loc[ht, base] - safe.loc[at, base]
            corrected.loc[mask, field] = adjusted
            if not np.isclose(original[field], adjusted):
                changed_features += 1
    if corrected.shape != reg.shape or not corrected.match_id.equals(reg.match_id):
        raise AssertionError("Unexpected dataset shape or row ordering")
    # Never modify labels, target goals, metadata, unaffected games or other predictors.
    unaffected = ~reg.match_id.isin(ids)
    if not corrected.loc[unaffected].equals(reg.loc[unaffected]):
        raise AssertionError("Unrelated match was changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_path, index=False)
    print(f"Cutoff: {CUTOFF}; excluded {len(late)} tournament team-records from historical baselines")
    print(f"Affected match IDs: {ids}")
    print(f"Adjusted {changed_features} field values in {len(ids)} regression matches; other rows unchanged")
    print(f"Saved distinct corrected dataset to {output_path}")
    return corrected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression", type=Path, default=Path("regression_2_1_dataset.csv"))
    parser.add_argument("--pre-matches", type=Path, default=Path("pre_tournament_matches.csv"))
    parser.add_argument("--baselines", type=Path, default=Path("pre_tournament_baselines.csv"))
    parser.add_argument("--master", type=Path, default=Path("world_cup_2026_master_matches.csv"))
    parser.add_argument("--output", type=Path, default=Path("regression_2_1_dataset_corrected.csv"))
    args = parser.parse_args()
    correction(args.regression, args.pre_matches, args.baselines, args.master, args.output)
