"""
app.py  —  World Cup 2026 Predictor  (full version with Tournament Tracker)
Run: streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone

from src.simulator import WorldCupSimulator, GROUPS, TEAM_ELO
from src.tournament_data import load_cache, refresh_cache, CACHE_FILE

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
)

# ── Load simulator ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_sim():
    return WorldCupSimulator()

@st.cache_data(ttl=300)  # cache tournament data for 5 minutes
def get_tournament_data():
    return load_cache()

sim  = get_sim()
data = get_tournament_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏆 FIFA World Cup 2026 Predictor")

live_path = PROC / "live_team_stats.json"
col_status, col_btn = st.columns([4, 1])
with col_status:
    if live_path.exists():
        with open(live_path) as f:
            lm = json.load(f)
        updated = lm.get("updated_at","")[:16].replace("T"," ")
        st.success(f"🟢 Live model data active · {lm.get('matches_processed',0)} matches · Updated {updated}")
    else:
        st.warning("⚠️  Run `python src/live_data.py` to load live match data into the model.")

with col_btn:
    if st.button("🔄 Refresh all data"):
        refresh_cache()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.caption("XGBoost · Monte Carlo simulation · Live data via football-data.org · Brikena Shaqi · SEEU 2026")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏟️ Tournament Tracker",
    "🎯 Remaining Matches",
    "🎲 Simulate Tournament",
    "⚽ Match Predictor",
    "📋 Groups & Teams",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOURNAMENT TRACKER (new!)
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🏟️ Live Tournament Tracker")

    updated_at = data.get("updated_at","")[:16].replace("T"," ")
    st.caption(
        f"Data updated: {updated_at} UTC · "
        f"{data.get('matches_finished',0)} matches played · "
        f"{data.get('matches_scheduled',0)} remaining"
    )

    # ── Live matches right now ────────────────────────────────────────────────
    live_now = data.get("matches",{}).get("live",[])
    if live_now:
        st.markdown("### 🔴 LIVE NOW")
        for m in live_now:
            c1,c2,c3 = st.columns([3,1,3])
            with c1: st.markdown(f"### {m['home']}")
            with c2: st.markdown(f"<h2 style='text-align:center'>{m['score_home']} — {m['score_away']}</h2>", unsafe_allow_html=True)
            with c3: st.markdown(f"### {m['away']}")
        st.divider()

    tracker_tab = st.tabs(["📊 Group Standings", "🏆 Bracket", "⚽ All Matches", "👟 Top Scorers"])

    # ── Group Standings ───────────────────────────────────────────────────────
    with tracker_tab[0]:
        st.markdown("### Group Stage Standings")
        standings = data.get("standings", {})

        if not standings:
            st.info("Group standings not available yet. Run `python src/tournament_data.py` to fetch.")
        else:
            groups_sorted = sorted(standings.keys())
            # Show 3 groups per row
            for row_start in range(0, len(groups_sorted), 3):
                cols = st.columns(3)
                for ci, g in enumerate(groups_sorted[row_start:row_start+3]):
                    table = standings[g]
                    with cols[ci]:
                        st.markdown(f"**Group {g}**")
                        rows = []
                        for t in table:
                            form_icons = ""
                            for r in (t.get("form") or "").split(","):
                                if r == "W": form_icons += "🟢"
                                elif r == "D": form_icons += "🟡"
                                elif r == "L": form_icons += "🔴"
                            rows.append({
                                "#":   t["position"],
                                "Team": t["short"] or t["team"],
                                "P":   t["played"],
                                "W":   t["won"],
                                "D":   t["draw"],
                                "L":   t["lost"],
                                "GD":  t["gd"],
                                "Pts": t["points"],
                                "Form": form_icons,
                            })
                        df = pd.DataFrame(rows).set_index("#")
                        st.dataframe(df, use_container_width=True, height=180)

    # ── Bracket ───────────────────────────────────────────────────────────────
    with tracker_tab[1]:
        st.markdown("### 🏆 Road to the Final")
        bracket = data.get("bracket", {})

        if not bracket:
            st.info("Knockout bracket not available yet — group stage still in progress.")
        else:
            stage_order = ["Round of 16", "Quarter-finals", "Semi-finals", "Third Place", "Final"]
            for stage in stage_order:
                matches_in_stage = bracket.get(stage, [])
                if not matches_in_stage:
                    continue
                st.markdown(f"#### {stage}")
                for m in matches_in_stage:
                    c1, c2, c3, c4 = st.columns([3, 1, 3, 2])
                    with c1:
                        st.markdown(f"**{m['home']}**")
                    with c2:
                        if m["status"] == "FINISHED":
                            st.markdown(f"**{m['score_home']} — {m['score_away']}**")
                        else:
                            st.markdown(f"*{m['date']}*")
                    with c3:
                        st.markdown(f"**{m['away']}**")
                    with c4:
                        if m["status"] == "FINISHED":
                            winner = m.get("winner")
                            if winner == "HOME_TEAM": st.success(f"✅ {m['home']}")
                            elif winner == "AWAY_TEAM": st.success(f"✅ {m['away']}")
                            else: st.info("Draw / Pens")
                        else:
                            st.caption(m.get("venue",""))
                st.divider()

    # ── All Matches ───────────────────────────────────────────────────────────
    with tracker_tab[2]:
        st.markdown("### All Matches")

        filter_status = st.radio(
            "Show",
            ["All", "Finished", "Upcoming"],
            horizontal=True,
        )

        all_matches = (
            data.get("matches",{}).get("finished",[]) +
            data.get("matches",{}).get("live",[]) +
            data.get("matches",{}).get("scheduled",[])
        )

        if filter_status == "Finished":
            all_matches = data.get("matches",{}).get("finished",[])
        elif filter_status == "Upcoming":
            all_matches = data.get("matches",{}).get("scheduled",[])

        if not all_matches:
            st.info("No matches to show.")
        else:
            rows = []
            for m in all_matches:
                score = (
                    f"{m['score_home']} — {m['score_away']}"
                    if m["status"] == "FINISHED"
                    else "vs"
                )
                rows.append({
                    "Date":   m["date"],
                    "Stage":  m["stage"].replace("_"," ").title(),
                    "Group":  (m.get("group") or "").replace("GROUP_",""),
                    "Home":   m["home"],
                    "Score":  score,
                    "Away":   m["away"],
                    "Venue":  m.get("venue",""),
                    "Status": m["status"],
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                height=500,
            )

    # ── Top Scorers ───────────────────────────────────────────────────────────
    with tracker_tab[3]:
        st.markdown("### 👟 Top Scorers — World Cup 2026")
        scorers = data.get("scorers", [])

        if not scorers:
            st.info("Scorer data not available yet.")
        else:
            rows = []
            for i, s in enumerate(scorers, 1):
                rows.append({
                    "#":          i,
                    "Player":     s["name"],
                    "Team":       s["team"],
                    "Nationality":s["nationality"],
                    "Goals":      s["goals"],
                    "Assists":    s["assists"],
                    "Penalties":  s["penalties"],
                    "Games":      s["played"],
                    "Goals/Game": round(s["goals"]/max(s["played"],1), 2),
                })

            df = pd.DataFrame(rows).set_index("#")
            st.dataframe(df, use_container_width=True, height=600)

            # Bar chart
            st.markdown("#### Goals scored")
            chart = pd.DataFrame(rows).head(15).set_index("Player")["Goals"]
            st.bar_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — REMAINING MATCHES
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎯 Remaining Matches — Predictions")
    st.markdown("Win probabilities based on current tournament form.")

    # Use live scheduled matches from API if available, else hardcoded
    scheduled = data.get("matches",{}).get("scheduled",[])
    knockout_stages = {"ROUND_OF_16","QUARTER_FINALS","SEMI_FINALS","FINAL","THIRD_PLACE"}

    upcoming_knockout = [m for m in scheduled if m.get("stage","") in knockout_stages]

    if upcoming_knockout:
        stage_order = {
            "ROUND_OF_16": 1, "QUARTER_FINALS": 2,
            "SEMI_FINALS": 3, "THIRD_PLACE": 4, "FINAL": 5
        }
        upcoming_knockout.sort(key=lambda m: (
            stage_order.get(m.get("stage",""), 9), m.get("date","")
        ))

        current_stage = None
        for match in upcoming_knockout:
            stage = match.get("stage","").replace("_"," ").title()
            if stage != current_stage:
                st.markdown(f"### {stage}")
                current_stage = stage

            home, away = match["home"], match["away"]

            try:
                ph, pd_, pa = sim.predict_proba(home, away, neutral=True)
            except Exception:
                ph, pd_, pa = 0.4, 0.2, 0.4

            with st.container():
                c_info, c_pred = st.columns([2, 4])
                with c_info:
                    st.markdown(f"**{home} vs {away}**")
                    st.caption(f"📅 {match['date']}  ·  📍 {match.get('venue','')}")
                with c_pred:
                    c1, c2, c3 = st.columns(3)
                    fav = max(ph, pd_, pa)
                    with c1:
                        st.metric(home, f"{ph*100:.0f}%", "⭐" if ph==fav else "")
                        st.progress(ph)
                    with c2:
                        st.metric("Draw", f"{pd_*100:.0f}%")
                        st.progress(pd_)
                    with c3:
                        st.metric(away, f"{pa*100:.0f}%", "⭐" if pa==fav else "")
                        st.progress(pa)
            st.divider()
    else:
        # Fallback hardcoded if API has no scheduled matches yet
        st.info("Fetching upcoming matches from API... If this persists, run `python src/tournament_data.py`")
        hardcoded = [
            {"stage": "Round of 16", "date": "Jul 5",  "home": "Norway",    "away": "Brazil",        "venue": "New York"},
            {"stage": "Round of 16", "date": "Jul 5",  "home": "Portugal",  "away": "Spain",         "venue": "Dallas"},
            {"stage": "Round of 16", "date": "Jul 6",  "home": "England",   "away": "Mexico",        "venue": "Los Angeles"},
            {"stage": "Round of 16", "date": "Jul 7",  "home": "Belgium",   "away": "United States", "venue": "Seattle"},
            {"stage": "Round of 16", "date": "Jul 7",  "home": "Argentina", "away": "Egypt",         "venue": "Houston"},
            {"stage": "Round of 16", "date": "Jul 8",  "home": "Colombia",  "away": "Switzerland",   "venue": "Miami"},
        ]
        for m in hardcoded:
            home, away = m["home"], m["away"]
            ph, pd_, pa = sim.predict_proba(home, away, neutral=True)
            fav = max(ph, pd_, pa)
            st.markdown(f"**{m['stage']} · {m['date']} · {m['venue']}**")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric(home, f"{ph*100:.0f}%", "⭐" if ph==fav else ""); st.progress(ph)
            with c2: st.metric("Draw", f"{pd_*100:.0f}%"); st.progress(pd_)
            with c3: st.metric(away, f"{pa*100:.0f}%", "⭐" if pa==fav else ""); st.progress(pa)
            st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎲 Tournament Simulation")
    st.write("Simulate the full 2026 World Cup thousands of times.")

    col_l, _ = st.columns([1, 3])
    with col_l:
        n_sims = st.selectbox("Simulations", [1000, 5000, 10000], index=1)
        run    = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run:
        bar  = st.progress(0, text="Simulating tournaments...")
        wins = {}
        batch = max(n_sims // 20, 1)
        for i in range(n_sims):
            w = sim.simulate_tournament()
            wins[w] = wins.get(w, 0) + 1
            if (i+1) % batch == 0:
                bar.progress((i+1)/n_sims, text=f"{i+1:,} / {n_sims:,}")
        bar.empty()

        results = pd.DataFrame([
            {"Team": t, "Win Probability": round(c/n_sims*100,1), "Simulated Wins": c}
            for t, c in wins.items()
        ]).sort_values("Win Probability", ascending=False).reset_index(drop=True)
        results.index += 1

        st.success(f"Done — {n_sims:,} tournaments simulated.")
        top3 = results.head(3)
        c1,c2,c3 = st.columns(3)
        for col, (_, row), medal in zip([c1,c2,c3], top3.iterrows(), ["🥇","🥈","🥉"]):
            with col:
                st.metric(f"{medal} {row['Team']}", f"{row['Win Probability']}%",
                          f"{row['Simulated Wins']:,} wins")
        st.bar_chart(results.head(16).set_index("Team")["Win Probability"], use_container_width=True)
        with st.expander("Full table"):
            st.dataframe(results, use_container_width=True, height=500)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MATCH PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("⚽ Match Predictor")
    all_teams = sorted(TEAM_ELO.keys())
    c1, c2, c3 = st.columns([5,1,5])
    with c1:
        home = st.selectbox("Home Team", all_teams, index=all_teams.index("Norway"))
    with c2:
        st.markdown("<br><h3 style='text-align:center;color:gray'>vs</h3>", unsafe_allow_html=True)
        neutral = st.checkbox("Neutral venue", value=True)
    with c3:
        away = st.selectbox("Away Team", all_teams, index=all_teams.index("Brazil"))

    if home == away:
        st.warning("Select two different teams.")
    elif st.button("⚽ Predict Match", type="primary"):
        ph, pd_, pa = sim.predict_proba(home, away, neutral=neutral)
        st.divider()
        fav = max(ph, pd_, pa)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric(f"🏠 {home} Win", f"{ph*100:.1f}%", "⭐ Favourite" if ph==fav else ""); st.progress(ph)
        with c2: st.metric("🤝 Draw", f"{pd_*100:.1f}%", "⭐ Favourite" if pd_==fav else ""); st.progress(pd_)
        with c3: st.metric(f"✈️ {away} Win", f"{pa*100:.1f}%", "⭐ Favourite" if pa==fav else ""); st.progress(pa)

        if fav==ph:   verdict = f"**{home}** are favoured"
        elif fav==pa: verdict = f"**{away}** are favoured"
        else:         verdict = "A **draw** is most likely"
        st.info(f"Most likely: {verdict} ({fav*100:.1f}%)")

        st.divider()
        st.subheader("Quick Simulation — 1,000 matches")
        hw = dw = aw = 0
        for _ in range(1000):
            r = sim.simulate_match(home, away, neutral=neutral, knockout=False)
            if r == home:     hw += 1
            elif r == "Draw": dw += 1
            else:             aw += 1
        h2h = pd.DataFrame({
            "Outcome":      [f"{home} Win","Draw",f"{away} Win"],
            "Out of 1,000": [hw, dw, aw],
            "%":            [f"{hw/10:.1f}%",f"{dw/10:.1f}%",f"{aw/10:.1f}%"],
        }).set_index("Outcome")
        st.dataframe(h2h, use_container_width=True)

        st.divider()
        st.subheader("Team Comparison")
        hs  = sim.team_stats.get(home,{})
        as_ = sim.team_stats.get(away,{})
        comp = pd.DataFrame({
            "Stat": ["Elo Rating","Form","Avg Goals","Avg Conceded","Data","WC Games"],
            home:   [TEAM_ELO.get(home,"N/A"), f"{hs.get('form',0):.3f}",
                     f"{hs.get('avg_goals',0):.2f}", f"{hs.get('avg_conceded',0):.2f}",
                     "🟢 live" if hs.get("live") else "📚 historical",
                     hs.get("tournament_games",0)],
            away:   [TEAM_ELO.get(away,"N/A"), f"{as_.get('form',0):.3f}",
                     f"{as_.get('avg_goals',0):.2f}", f"{as_.get('avg_conceded',0):.2f}",
                     "🟢 live" if as_.get("live") else "📚 historical",
                     as_.get("tournament_games",0)],
        }).set_index("Stat")
        st.dataframe(comp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — GROUPS & TEAMS
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("2026 World Cup — Official Groups")
    cols = st.columns(4)
    for i, (g, teams) in enumerate(sorted(GROUPS.items())):
        with cols[i % 4]:
            st.markdown(f"**Group {g}**")
            for team in teams:
                elo  = TEAM_ELO.get(team,"?")
                live = sim.team_stats.get(team,{}).get("live", False)
                st.markdown(f"- {team} `{elo}` {'🟢' if live else ''}")
            st.markdown("")

    st.divider()
    st.subheader("All Teams — Stats")
    rows = []
    for team in sorted(TEAM_ELO.keys()):
        s = sim.team_stats.get(team,{})
        rows.append({
            "Team":         team,
            "Elo":          TEAM_ELO.get(team,"N/A"),
            "Form":         round(s.get("form",0),3),
            "Avg Goals":    round(s.get("avg_goals",0),2),
            "Avg Conceded": round(s.get("avg_conceded",0),2),
            "WC Games":     s.get("tournament_games",0),
            "Data":         "🟢" if s.get("live") else "📚",
        })
    st.dataframe(
        pd.DataFrame(rows).sort_values("Elo",ascending=False).reset_index(drop=True),
        use_container_width=True, height=600,
    )

st.divider()
st.caption("Built with XGBoost + Monte Carlo · Live data via football-data.org · Brikena Shaqi · SEEU 2026")