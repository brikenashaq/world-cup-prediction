"""
src/tournament_data.py
Fetches live World Cup 2026 data: standings, scorers, bracket, matches.
All data cached to data/processed/tournament_cache.json
Run standalone: python src/tournament_data.py
Or imported by app.py for live display.
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT  = Path(__file__).parent.parent
PROC  = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CACHE_FILE = PROC / "tournament_cache.json"
API_KEY    = "10b06a04a02f483884448c688bd3f12f"
BASE       = "https://api.football-data.org/v4"
HEADERS    = {"X-Auth-Token": API_KEY}

# Rate limit: free tier = 10 requests/minute
def fetch(endpoint: str, params: dict = None, retries: int = 2) -> dict:
    url = f"{BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f"  Rate limited — waiting 15s...")
                time.sleep(15)
            else:
                print(f"  ⚠️ {r.status_code} on {endpoint}: {r.text[:100]}")
                return {}
        except Exception as e:
            print(f"  ⚠️ Request failed: {e}")
    return {}


# ── Fetch functions ───────────────────────────────────────────────────────────

def fetch_standings() -> list:
    """Group stage standings — one table per group."""
    print("  Fetching group standings...")
    data = fetch("competitions/WC/standings", {"season": "2026"})
    return data.get("standings", [])


def fetch_matches() -> list:
    """All WC 2026 matches — finished + scheduled."""
    print("  Fetching all matches...")
    data = fetch("competitions/WC/matches", {"season": "2026"})
    return data.get("matches", [])


def fetch_scorers(limit: int = 30) -> list:
    """Top scorers in WC 2026."""
    print("  Fetching top scorers...")
    data = fetch("competitions/WC/scorers", {"season": "2026", "limit": limit})
    return data.get("scorers", [])


def fetch_teams() -> list:
    """All teams with crest URLs."""
    print("  Fetching teams...")
    data = fetch("competitions/WC/teams", {"season": "2026"})
    return data.get("teams", [])


# ── Process data ──────────────────────────────────────────────────────────────

def process_standings(raw: list) -> dict:
    """
    Returns dict: { "A": [{"pos":1,"team":"France","played":3,...},...], ... }
    """
    groups = {}
    for standing in raw:
        group_name = standing.get("group", "")
        if not group_name:
            continue
        # e.g. "GROUP_A" -> "A"
        label = group_name.replace("GROUP_", "").replace("Group ", "")
        table = []
        for row in standing.get("table", []):
            team = row.get("team", {})
            table.append({
                "position":   row.get("position"),
                "team":       team.get("name", ""),
                "short":      team.get("shortName", ""),
                "crest":      team.get("crest", ""),
                "played":     row.get("playedGames", 0),
                "won":        row.get("won", 0),
                "draw":       row.get("draw", 0),
                "lost":       row.get("lost", 0),
                "gf":         row.get("goalsFor", 0),
                "ga":         row.get("goalsAgainst", 0),
                "gd":         row.get("goalDifference", 0),
                "points":     row.get("points", 0),
                "form":       row.get("form", ""),
            })
        groups[label] = table
    return groups


def process_matches(raw: list) -> dict:
    """
    Returns dict with keys: finished, scheduled, live, by_stage
    """
    finished  = []
    scheduled = []
    live      = []
    by_stage  = {}

    for m in raw:
        status = m.get("status", "")
        stage  = m.get("stage", "UNKNOWN")
        home   = m.get("homeTeam", {}).get("name", "?")
        away   = m.get("awayTeam", {}).get("name", "?")
        score  = m.get("score", {})
        ft     = score.get("fullTime", {})
        ht     = score.get("halfTime", {})
        date   = m.get("utcDate", "")[:10]
        time_  = m.get("utcDate", "")[11:16]
        venue  = m.get("venue", "")
        group  = m.get("group", "")

        entry = {
            "id":       m.get("id"),
            "date":     date,
            "time":     time_,
            "stage":    stage,
            "group":    group,
            "home":     home,
            "away":     away,
            "home_crest": m.get("homeTeam", {}).get("crest", ""),
            "away_crest": m.get("awayTeam", {}).get("crest", ""),
            "score_home": ft.get("home"),
            "score_away": ft.get("away"),
            "ht_home":  ht.get("home"),
            "ht_away":  ht.get("away"),
            "venue":    venue,
            "status":   status,
            "winner":   score.get("winner"),
        }

        if status == "FINISHED":
            finished.append(entry)
        elif status in ("IN_PLAY", "PAUSED", "HALFTIME"):
            live.append(entry)
        else:
            scheduled.append(entry)

        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(entry)

    return {
        "finished":  finished,
        "scheduled": scheduled,
        "live":      live,
        "by_stage":  by_stage,
        "total":     len(raw),
    }


def process_scorers(raw: list) -> list:
    """Returns clean list of top scorers."""
    result = []
    for s in raw:
        player = s.get("player", {})
        team   = s.get("team", {})
        result.append({
            "name":        player.get("name", ""),
            "nationality": player.get("nationality", ""),
            "team":        team.get("name", ""),
            "team_crest":  team.get("crest", ""),
            "goals":       s.get("goals", 0),
            "assists":     s.get("assists", 0),
            "penalties":   s.get("penalties", 0),
            "played":      s.get("playedMatches", 0),
        })
    return sorted(result, key=lambda x: (x["goals"] or 0, x["assists"] or 0), reverse=True)


def build_bracket(matches: dict) -> dict:
    """
    Builds the knockout bracket from match data.
    Returns dict with rounds and their matches.
    """
    bracket = {}
    knockout_stages = [
        "ROUND_OF_16",
        "QUARTER_FINALS",
        "SEMI_FINALS",
        "THIRD_PLACE",
        "FINAL",
    ]
    stage_labels = {
        "ROUND_OF_16":   "Round of 16",
        "QUARTER_FINALS":"Quarter-finals",
        "SEMI_FINALS":   "Semi-finals",
        "THIRD_PLACE":   "Third Place",
        "FINAL":         "Final",
    }
    for stage in knockout_stages:
        stage_matches = matches["by_stage"].get(stage, [])
        if stage_matches:
            bracket[stage_labels.get(stage, stage)] = stage_matches
    return bracket


# ── Main cache builder ────────────────────────────────────────────────────────

def refresh_cache() -> dict:
    """Fetches all data and saves to cache file. Returns the cache dict."""
    print("=" * 50)
    print("🔄 Refreshing tournament data cache...")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    raw_standings = fetch_standings()
    time.sleep(1)  # respect rate limit
    raw_matches   = fetch_matches()
    time.sleep(1)
    raw_scorers   = fetch_scorers(30)

    standings = process_standings(raw_standings)
    matches   = process_matches(raw_matches)
    scorers   = process_scorers(raw_scorers)
    bracket   = build_bracket(matches)

    cache = {
        "updated_at":         datetime.now(timezone.utc).isoformat(),
        "matches_finished":   len(matches["finished"]),
        "matches_scheduled":  len(matches["scheduled"]),
        "matches_live":       len(matches["live"]),
        "standings":          standings,
        "matches":            matches,
        "scorers":            scorers,
        "bracket":            bracket,
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)

    print(f"\n✅ Cache saved → {CACHE_FILE}")
    print(f"   Finished matches: {cache['matches_finished']}")
    print(f"   Scheduled:        {cache['matches_scheduled']}")
    print(f"   Live now:         {cache['matches_live']}")
    print(f"   Groups in standings: {list(standings.keys())}")
    print(f"   Top scorer: {scorers[0]['name']} ({scorers[0]['goals']} goals)" if scorers else "   No scorers yet")
    print(f"   Bracket rounds: {list(bracket.keys())}")

    return cache


def load_cache() -> dict:
    """Load cache from disk. Refresh if missing."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return refresh_cache()


if __name__ == "__main__":
    cache = refresh_cache()