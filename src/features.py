"""
src/features.py  —  Build feature table from raw CSVs
Run: python src/features.py
Out: data/processed/features.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "FIFA World Cup":               1.00,
    "UEFA Euro":                    0.90,
    "Copa América":                 0.90,
    "Africa Cup of Nations":        0.85,
    "AFC Asian Cup":                0.85,
    "CONCACAF Gold Cup":            0.80,
    "UEFA Nations League":          0.75,
    "FIFA World Cup qualification": 0.75,
    "Friendly":                     0.40,
}

def get_weight(t):
    for k, v in WEIGHTS.items():
        if k in str(t): return v
    return 0.60

def get_result(row):
    if   row["home_score"] > row["away_score"]: return "Home Win"
    elif row["home_score"] < row["away_score"]: return "Away Win"
    else:                                       return "Draw"

def main():
    print("Loading raw data...")
    matches = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    elo_raw = pd.read_csv(RAW / "eloratings.csv")
    elo_raw["date"] = pd.to_datetime(elo_raw["date"], format="mixed")

    matches = matches[matches["date"] >= "1993-01-01"].copy()
    matches = matches.dropna(subset=["home_score","away_score"])
    matches = matches.sort_values("date").reset_index(drop=True)
    matches["result"] = matches.apply(get_result, axis=1)
    matches["weight"] = matches["tournament"].apply(get_weight)
    print(f"  Matches after filter: {len(matches)}")

    # Rolling form features
    home = matches[["date","home_team","home_score","away_score","result","weight"]].copy()
    home.columns = ["date","team","scored","conceded","result","weight"]
    home["win"]  = (home["result"] == "Home Win").astype(float)
    home["draw"] = (home["result"] == "Draw").astype(float)

    away = matches[["date","away_team","away_score","home_score","result","weight"]].copy()
    away.columns = ["date","team","scored","conceded","result","weight"]
    away["win"]  = (away["result"] == "Away Win").astype(float)
    away["draw"] = (away["result"] == "Draw").astype(float)

    long = pd.concat([home, away]).sort_values(["team","date"]).reset_index(drop=True)
    long["points"] = (long["win"] * 3 + long["draw"]) * long["weight"]

    for col in ["points","scored","conceded"]:
        long[f"roll_{col}"] = (
            long.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )

    team_stats = long[["date","team","roll_points","roll_scored","roll_conceded"]].copy()

    # Elo merge (pre-match rating using merge_asof backward)
    print("  Merging Elo ratings...")
    elo_raw = elo_raw.sort_values("date")
    matches  = matches.sort_values("date")

    for side in ["home","away"]:
        col = f"{side}_team"
        elo = elo_raw.rename(columns={"team": col, "rating": f"{side}_elo"})[
            ["date", col, f"{side}_elo"]
        ]
        merged = pd.merge_asof(matches[["date",col]], elo,
                               on="date", by=col, direction="backward")
        matches[f"{side}_elo"] = merged[f"{side}_elo"].values

    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]

    # Merge rolling stats back
    for side in ["home","away"]:
        matches = matches.merge(
            team_stats.rename(columns={
                "team":          f"{side}_team",
                "roll_points":   f"{side}_form",
                "roll_scored":   f"{side}_avg_goals",
                "roll_conceded": f"{side}_avg_conceded",
            }),
            on=["date", f"{side}_team"], how="left"
        )

    KEEP = ["date","home_team","away_team",
            "elo_diff","home_elo","away_elo",
            "home_form","away_form",
            "home_avg_goals","away_avg_goals",
            "home_avg_conceded","away_avg_conceded",
            "neutral","result"]

    DROP_NAN = [c for c in KEEP if c not in ("date","home_team","away_team","result","neutral")]
    final = matches[KEEP].dropna(subset=DROP_NAN).reset_index(drop=True)
    final.to_csv(PROC / "features.csv", index=False)
    print(f"  ✅ Saved features.csv  shape={final.shape}")

if __name__ == "__main__":
    main()