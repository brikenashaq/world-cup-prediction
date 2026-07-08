"""
src/simulator.py  —  World Cup 2026 Simulator
Automatically uses live tournament stats when available.
"""
import json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
MDIR = ROOT / "models"
PROC = ROOT / "data" / "processed"

# ── Official 2026 World Cup Groups ────────────────────────────────────────────
GROUPS = {
    "A": ["Mexico",        "South Africa",           "South Korea",   "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",                "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",               "Australia",     "Turkey"],
    "E": ["Germany",       "Curaçao",                "Ivory Coast",   "Ecuador"],
    "F": ["Netherlands",   "Japan",                  "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                  "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",             "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",                "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",                "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",               "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",                "Ghana",         "Panama"],
}

# ── Team Elo ratings (July 2026) ──────────────────────────────────────────────
TEAM_ELO = {
    "France":        2010, "Spain":         2000, "England":       1975,
    "Brazil":        1970, "Argentina":     1965, "Portugal":      1960,
    "Germany":       1948, "Netherlands":   1940, "Belgium":       1920,
    "Norway":        1910, "Italy":         1915, "Croatia":       1890,
    "Uruguay":       1885, "Colombia":      1880, "Mexico":        1870,
    "Switzerland":   1858, "Japan":         1850, "United States": 1845,
    "South Korea":   1840, "Senegal":       1835, "Morocco":       1835,
    "Ecuador":       1820, "Serbia":        1815, "Australia":     1800,
    "Nigeria":       1795, "Chile":         1790, "Peru":          1780,
    "Algeria":       1775, "Tunisia":       1770, "Iran":          1760,
    "Saudi Arabia":  1755, "Venezuela":     1740, "Canada":        1735,
    "Egypt":         1730, "Mali":          1725, "Cameroon":      1720,
    "Bolivia":       1710, "Paraguay":      1710, "DR Congo":      1700,
    "Jamaica":       1680, "Iraq":          1675, "Panama":        1660,
    "Sweden":        1650, "New Zealand":   1640, "Uzbekistan":    1620,
    "Jordan":        1610, "Austria":       1605, "Haiti":         1590,
    "Cape Verde":    1585, "South Africa":  1560, "Ivory Coast":   1555,
    "Scotland":      1550, "Ghana":         1545, "Czech Republic":1540,
    "Qatar":         1520, "Turkey":        1530, "Curaçao":       1510,
    "Bosnia and Herzegovina": 1640,
}


class WorldCupSimulator:

    def __init__(self):
        self.model   = joblib.load(MDIR / "model.pkl")
        self.le      = joblib.load(MDIR / "label_encoder.pkl")
        with open(MDIR / "metadata.json") as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.classes      = meta["classes"]
        self.i_home  = self.classes.index("Home Win")
        self.i_draw  = self.classes.index("Draw")
        self.i_away  = self.classes.index("Away Win")
        self.live_updated_at = None
        self.team_stats = self._build_team_stats()

    # ── Build team stats: live if available, else historical ─────────────────
    def _build_team_stats(self) -> dict:
        live_path = PROC / "live_team_stats.json"

        # Start with historical baseline
        stats = self._build_historical_stats()

        # Override with live blended stats if file exists
        if live_path.exists():
            with open(live_path) as f:
                live_data = json.load(f)

            self.live_updated_at = live_data.get("updated_at")
            live_teams = live_data.get("teams", {})

            print(f"  ✅ Live data loaded ({len(live_teams)} teams, "
                  f"updated {self.live_updated_at[:16]})")

            for team, live_stats in live_teams.items():
                if team in stats:
                    # Merge: keep elo from historical, update form/goals from live
                    stats[team]["form"]         = live_stats["form"]
                    stats[team]["avg_goals"]    = live_stats["avg_goals"]
                    stats[team]["avg_conceded"] = live_stats["avg_conceded"]
                    stats[team]["live"]         = True
                    stats[team]["tournament_games"] = live_stats.get("tournament_games", 0)
                else:
                    stats[team] = {
                        "form":             live_stats["form"],
                        "avg_goals":        live_stats["avg_goals"],
                        "avg_conceded":     live_stats["avg_conceded"],
                        "elo":              TEAM_ELO.get(team, 1700),
                        "live":             True,
                        "tournament_games": live_stats.get("tournament_games", 0),
                    }
        else:
            print("  ℹ️  No live data found — using historical stats.")
            print("     Run: python src/live_data.py  to fetch live data.")

        return stats

    def _build_historical_stats(self) -> dict:
        df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
        df = df.dropna().sort_values("date")
        stats = {}

        all_teams = set(df["home_team"]) | set(df["away_team"])
        for team in all_teams:
            h = df[df["home_team"] == team][
                ["date","home_form","home_avg_goals","home_avg_conceded"]
            ].rename(columns={
                "home_form": "form",
                "home_avg_goals": "avg_goals",
                "home_avg_conceded": "avg_conceded",
            })
            a = df[df["away_team"] == team][
                ["date","away_form","away_avg_goals","away_avg_conceded"]
            ].rename(columns={
                "away_form": "form",
                "away_avg_goals": "avg_goals",
                "away_avg_conceded": "avg_conceded",
            })
            combined = pd.concat([h, a]).sort_values("date")
            if len(combined) > 0:
                latest = combined.iloc[-1]
                stats[team] = {
                    "form":             float(latest["form"]),
                    "avg_goals":        float(latest["avg_goals"]),
                    "avg_conceded":     float(latest["avg_conceded"]),
                    "elo":              TEAM_ELO.get(team, 1700),
                    "live":             False,
                    "tournament_games": 0,
                }

        # Fill in any qualified teams not in historical data
        for team, elo in TEAM_ELO.items():
            if team not in stats:
                stats[team] = {
                    "form": 0.5, "avg_goals": 1.1,
                    "avg_conceded": 1.1, "elo": elo,
                    "live": False, "tournament_games": 0,
                }
        return stats

    def reload_live_stats(self):
        """Call this to refresh stats without restarting the app."""
        self.team_stats = self._build_team_stats()

    # ── Core prediction ───────────────────────────────────────────────────────
    def predict_proba(self, home: str, away: str, neutral: bool = True):
        """Returns (p_home_win, p_draw, p_away_win)"""
        hs  = self.team_stats.get(home, {})
        as_ = self.team_stats.get(away, {})
        row = {
            "elo_diff":          hs.get("elo", 1700) - as_.get("elo", 1700),
            "home_form":         hs.get("form", 0.5),
            "away_form":         as_.get("form", 0.5),
            "home_avg_goals":    hs.get("avg_goals", 1.1),
            "away_avg_goals":    as_.get("avg_goals", 1.1),
            "home_avg_conceded": hs.get("avg_conceded", 1.1),
            "away_avg_conceded": as_.get("avg_conceded", 1.1),
            "neutral":           int(neutral),
        }
        X = pd.DataFrame([row])[self.feature_cols]
        p = self.model.predict_proba(X)[0]
        return float(p[self.i_home]), float(p[self.i_draw]), float(p[self.i_away])

    def simulate_match(self, home: str, away: str,
                       neutral: bool = True, knockout: bool = False) -> str:
        ph, pd_, pa = self.predict_proba(home, away, neutral)
        if knockout:
            ph += pd_ * 0.50
            pa += pd_ * 0.50
            pd_ = 0.0
        outcome = np.random.choice(["H","D","A"], p=[ph, pd_, pa])
        if outcome == "H": return home
        if outcome == "A": return away
        return "Draw"

    # ── Group stage ───────────────────────────────────────────────────────────
    def simulate_group(self, teams: list) -> list:
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        gf  = {t: 0 for t in teams}

        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                t1, t2 = teams[i], teams[j]
                ph, pd_, pa = self.predict_proba(t1, t2, neutral=True)
                outcome = np.random.choice(["H","D","A"], p=[ph, pd_, pa])
                g1 = np.random.poisson(self.team_stats.get(t1,{}).get("avg_goals",1.1))
                g2 = np.random.poisson(self.team_stats.get(t2,{}).get("avg_goals",1.1))

                if outcome == "H":
                    pts[t1] += 3
                    g1 = max(g1, g2+1)
                elif outcome == "A":
                    pts[t2] += 3
                    g2 = max(g2, g1+1)
                else:
                    pts[t1] += 1
                    pts[t2] += 1
                    g1 = g2 = max(g1, g2)

                gd[t1] += g1 - g2
                gd[t2] += g2 - g1
                gf[t1] += g1
                gf[t2] += g2

        return sorted(teams,
                      key=lambda t: (pts[t], gd[t], gf[t]),
                      reverse=True)

    # ── Knockout ──────────────────────────────────────────────────────────────
    def simulate_knockout_bracket(self, teams: list) -> str:
        remaining = list(teams)
        while len(remaining) > 1:
            next_round = []
            for i in range(0, len(remaining), 2):
                if i+1 < len(remaining):
                    w = self.simulate_match(remaining[i], remaining[i+1],
                                            neutral=True, knockout=True)
                    next_round.append(w)
                else:
                    next_round.append(remaining[i])
            remaining = next_round
        return remaining[0]

    # ── Full tournament ───────────────────────────────────────────────────────
    def simulate_tournament(self) -> str:
        group_results = {}
        third_place   = []
        for g, teams in GROUPS.items():
            ranked = self.simulate_group(teams)
            group_results[g] = ranked
            third_place.append(ranked[2])

        r32 = []
        for g in sorted(GROUPS.keys()):
            r32.append(group_results[g][0])
            r32.append(group_results[g][1])

        best_thirds = sorted(third_place,
                             key=lambda t: TEAM_ELO.get(t, 1600),
                             reverse=True)[:8]
        r32.extend(best_thirds)
        np.random.shuffle(r32)
        return self.simulate_knockout_bracket(r32)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    def monte_carlo(self, n: int = 10000) -> pd.DataFrame:
        wins = {}
        for i in range(n):
            w = self.simulate_tournament()
            wins[w] = wins.get(w, 0) + 1
        return pd.DataFrame([
            {"team": t, "wins": c, "probability": round(c/n*100, 2)}
            for t, c in wins.items()
        ]).sort_values("probability", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    sim     = WorldCupSimulator()
    results = sim.monte_carlo(10000)
    print("\n🏆 World Cup 2026 — Win Probabilities")
    print(results.head(20).to_string(index=False))