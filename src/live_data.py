"""
src/live_data.py — Fetch live World Cup 2026 data and update team stats
Run: python src/live_data.py
Updates: data/processed/live_team_stats.json

This script:
1. Fetches all finished WC 2026 matches from football-data.org
2. Calculates tournament-specific form per team (goals, results, opponent strength)
3. Saves updated stats that the simulator uses automatically
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT  = Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── Your API key ──────────────────────────────────────────────────────────────
API_KEY = "10b06a04a02f483884448c688bd3f12f"
BASE    = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# ── Opponent strength map (Elo-based, for weighting results by opponent) ──────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import TEAM_ELO

def fetch(endpoint: str, params: dict = None):
    url = f"{BASE}/{endpoint}"
    r   = requests.get(url, headers=HEADERS, params=params, timeout=10)
    if r.status_code == 200:
        return r.json()
    print(f"  ⚠️  {r.status_code}: {r.text[:200]}")
    return None


def fetch_wc_matches() -> list:
    """Fetch all finished World Cup 2026 matches."""
    print("Fetching finished matches...")
    data = fetch("competitions/WC/matches", {"season": "2026", "status": "FINISHED"})
    if not data:
        return []
    matches = data.get("matches", [])
    print(f"  Found {len(matches)} finished matches")
    return matches


def normalise_name(name: str) -> str:
    """Normalise team names to match our TEAM_ELO keys."""
    mapping = {
        "USA":                        "United States",
        "United States of America":   "United States",
        "Korea Republic":             "South Korea",
        "IR Iran":                    "Iran",
        "Czechia":                    "Czech Republic",
        "Côte d'Ivoire":              "Ivory Coast",
        "Cote d'Ivoire":              "Ivory Coast",
        "Bosnia-Herzegovina":         "Bosnia and Herzegovina",
        "Türkiye":                    "Turkey",
        "Cape Verde Islands":         "Cape Verde",
        "DR Congo":                   "DR Congo",
        "Scotland":                   "Scotland",
    }
    return mapping.get(name, name)


def calculate_tournament_stats(matches: list) -> dict:
    """
    For each team compute:
    - tournament_form:       weighted points per game (weights by opponent Elo)
    - tournament_avg_goals:  avg goals scored per game
    - tournament_avg_conceded: avg goals conceded per game
    - tournament_games:      number of games played
    - tournament_results:    list of results (W/D/L) with opponent and score
    """
    team_records = {}

    for m in matches:
        home = normalise_name(m["homeTeam"]["name"])
        away = normalise_name(m["awayTeam"]["name"])
        score = m.get("score", {})
        ft    = score.get("fullTime", {})
        hg    = ft.get("home")
        ag    = ft.get("away")

        if hg is None or ag is None:
            continue

        hg, ag = int(hg), int(ag)

        # Opponent Elo — stronger opponent = result worth more
        home_opp_elo = TEAM_ELO.get(away, 1700)
        away_opp_elo = TEAM_ELO.get(home, 1700)

        # Weight = opponent_elo / 1800 (normalised so avg opponent = 1.0)
        home_weight = home_opp_elo / 1800
        away_weight = away_opp_elo / 1800

        date = m.get("utcDate", "")[:10]

        for team, scored, conceded, weight, opp in [
            (home, hg, ag, home_weight, away),
            (away, ag, hg, away_weight, home),
        ]:
            if team not in team_records:
                team_records[team] = {
                    "games":     0,
                    "points":    0.0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "results":   [],
                }

            r = team_records[team]
            r["games"]       += 1
            r["goals_for"]   += scored
            r["goals_against"] += conceded

            if scored > conceded:
                pts = 3 * weight
                result = "W"
            elif scored == conceded:
                pts = 1 * weight
                result = "D"
            else:
                pts = 0
                result = "L"

            r["points"] += pts
            r["results"].append({
                "date":     date,
                "opponent": opp,
                "scored":   scored,
                "conceded": conceded,
                "result":   result,
            })

    # Build final stats dict
    stats = {}
    for team, r in team_records.items():
        g = max(r["games"], 1)
        stats[team] = {
            "tournament_form":           round(r["points"] / g, 3),
            "tournament_avg_goals":      round(r["goals_for"] / g, 2),
            "tournament_avg_conceded":   round(r["goals_against"] / g, 2),
            "tournament_games":          r["games"],
            "results":                   r["results"],
            "total_goals":               r["goals_for"],
            "total_conceded":            r["goals_against"],
        }

    return stats


def blend_with_historical(
    live_stats: dict,
    historical_path: Path,
    live_weight: float = 0.65,
) -> dict:
    """
    Blend tournament stats with historical model stats.
    live_weight = how much to trust the tournament data vs historical.
    0.65 means tournament results count for 65% of the final rating.
    Increases as team plays more games (3 games = less trust than 6 games).
    """
    hist_df = pd.read_csv(historical_path, parse_dates=["date"])
    hist_df = hist_df.dropna().sort_values("date")

    blended = {}

    for team, live in live_stats.items():
        # Dynamic weight: more tournament games = trust live data more
        games  = live["tournament_games"]
        weight = min(0.4 + games * 0.08, 0.85)  # 3 games=64%, 6 games=88%

        # Get historical stats for this team
        h_rows = pd.concat([
            hist_df[hist_df["home_team"] == team][
                ["date","home_form","home_avg_goals","home_avg_conceded"]
            ].rename(columns={
                "home_form": "form",
                "home_avg_goals": "avg_goals",
                "home_avg_conceded": "avg_conceded"
            }),
            hist_df[hist_df["away_team"] == team][
                ["date","away_form","away_avg_goals","away_avg_conceded"]
            ].rename(columns={
                "away_form": "form",
                "away_avg_goals": "avg_goals",
                "away_avg_conceded": "avg_conceded"
            }),
        ]).sort_values("date")

        if len(h_rows) > 0:
            hist = h_rows.iloc[-1]
            hist_form     = float(hist["form"])
            hist_goals    = float(hist["avg_goals"])
            hist_conceded = float(hist["avg_conceded"])
        else:
            hist_form     = 0.5
            hist_goals    = 1.1
            hist_conceded = 1.1

        blended[team] = {
            "form":          round(weight * live["tournament_form"] +
                                   (1-weight) * hist_form, 3),
            "avg_goals":     round(weight * live["tournament_avg_goals"] +
                                   (1-weight) * hist_goals, 2),
            "avg_conceded":  round(weight * live["tournament_avg_conceded"] +
                                   (1-weight) * hist_conceded, 2),
            "tournament_games":    games,
            "tournament_results":  live["results"],
            "data_source":         "blended_live_historical",
            "live_weight":         round(weight, 2),
        }

    return blended


def main():
    print("=" * 50)
    print("World Cup 2026 — Live Data Updater")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    # 1. Fetch matches
    matches = fetch_wc_matches()
    if not matches:
        print("No matches fetched. Check API key or internet connection.")
        return

    # 2. Calculate tournament stats
    print("\nCalculating tournament stats...")
    live_stats = calculate_tournament_stats(matches)
    print(f"  Teams with tournament data: {len(live_stats)}")

    # Print summary
    print("\n📊 Tournament Performance Summary:")
    sorted_teams = sorted(
        live_stats.items(),
        key=lambda x: x[1]["tournament_form"],
        reverse=True
    )
    for team, s in sorted_teams[:16]:
        results_str = " ".join([r["result"] for r in s["results"]])
        print(f"  {team:<25} {results_str:<10} "
              f"form={s['tournament_form']:.2f}  "
              f"GF={s['total_goals']}  GA={s['total_conceded']}")

    # 3. Blend with historical
    print("\nBlending with historical stats...")
    blended = blend_with_historical(
        live_stats,
        historical_path=PROC / "features.csv",
    )

    # 4. Save
    out = PROC / "live_team_stats.json"
    with open(out, "w") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(),
            "matches_processed": len(matches),
            "teams": blended,
        }, f, indent=2)

    print(f"\n✅ Saved live stats for {len(blended)} teams → {out}")
    print("\nRestart Streamlit to apply updated stats.")


if __name__ == "__main__":
    main()