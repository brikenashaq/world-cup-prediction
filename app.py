"""
app.py  —  World Cup 2026 Predictor
Run: streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from src.simulator import WorldCupSimulator, GROUPS, TEAM_ELO
from src.tournament_data import load_cache, refresh_cache, CACHE_FILE

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def get_sim():
    return WorldCupSimulator()

@st.cache_data(ttl=300)
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏟️ Tournament Tracker",
    "🎯 Remaining Matches",
    "🎲 Simulate Tournament",
    "⚽ Match Predictor",
    "📺 Watch Live",
    "📋 Groups & Teams",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOURNAMENT TRACKER
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🏟️ Live Tournament Tracker")
    updated_at = data.get("updated_at","")[:16].replace("T"," ")
    st.caption(
        f"Data updated: {updated_at} UTC · "
        f"{data.get('matches_finished',0)} matches played · "
        f"{data.get('matches_scheduled',0)} remaining"
    )

    live_now = data.get("matches",{}).get("live",[])
    if live_now:
        st.markdown("### 🔴 LIVE NOW")
        for m in live_now:
            c1,c2,c3 = st.columns([3,1,3])
            with c1: st.markdown(f"### {m.get('home') or 'TBD'}")
            with c2: st.markdown(f"<h2 style='text-align:center'>{m['score_home']} — {m['score_away']}</h2>", unsafe_allow_html=True)
            with c3: st.markdown(f"### {m.get('away') or 'TBD'}")
        st.divider()

    tracker_tab = st.tabs(["📊 Group Standings", "🏆 Bracket", "⚽ All Matches", "👟 Top Scorers"])

    with tracker_tab[0]:
        st.markdown("### Group Stage Standings")
        standings = data.get("standings", {})
        if not standings:
            st.info("Group standings not available yet. Run `python src/tournament_data.py` to fetch.")
        else:
            groups_sorted = sorted(standings.keys())
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
                                if r == "W":   form_icons += "🟢"
                                elif r == "D": form_icons += "🟡"
                                elif r == "L": form_icons += "🔴"
                            rows.append({
                                "#":    t["position"],
                                "Team": t.get("short") or t.get("team",""),
                                "P":    t["played"],
                                "W":    t["won"],
                                "D":    t["draw"],
                                "L":    t["lost"],
                                "GD":   t["gd"],
                                "Pts":  t["points"],
                                "Form": form_icons,
                            })
                        st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True, height=180)

    with tracker_tab[1]:
        st.markdown("### 🏆 Road to the Final")
        bracket = data.get("bracket", {})
        if not bracket:
            st.info("Knockout bracket not available yet — group stage still in progress.")
        else:
            for stage in ["Round of 16","Quarter-finals","Semi-finals","Third Place","Final"]:
                stage_matches = bracket.get(stage, [])
                if not stage_matches:
                    continue
                st.markdown(f"#### {stage}")
                for m in stage_matches:
                    home_name = m.get("home") or "TBD"
                    away_name = m.get("away") or "TBD"
                    c1,c2,c3,c4 = st.columns([3,1,3,2])
                    with c1: st.markdown(f"**{home_name}**")
                    with c2:
                        if m["status"] == "FINISHED":
                            st.markdown(f"**{m['score_home']} — {m['score_away']}**")
                        else:
                            st.markdown(f"*{m['date']}*")
                    with c3: st.markdown(f"**{away_name}**")
                    with c4:
                        if m["status"] == "FINISHED":
                            winner = m.get("winner")
                            if winner == "HOME_TEAM":   st.success(f"✅ {home_name}")
                            elif winner == "AWAY_TEAM": st.success(f"✅ {away_name}")
                            else:                       st.info("Pens")
                        else:
                            st.caption(m.get("venue",""))
                st.divider()

    with tracker_tab[2]:
        st.markdown("### All Matches")
        filter_status = st.radio("Show", ["All","Finished","Upcoming"], horizontal=True)
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
                score = f"{m['score_home']} — {m['score_away']}" if m["status"] == "FINISHED" else "vs"
                rows.append({
                    "Date":   m["date"],
                    "Stage":  m["stage"].replace("_"," ").title(),
                    "Group":  (m.get("group") or "").replace("GROUP_",""),
                    "Home":   m.get("home") or "TBD",
                    "Score":  score,
                    "Away":   m.get("away") or "TBD",
                    "Venue":  m.get("venue",""),
                    "Status": m["status"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500)

    with tracker_tab[3]:
        st.markdown("### 👟 Top Scorers — World Cup 2026")
        scorers = data.get("scorers", [])
        if not scorers:
            st.info("Scorer data not available yet.")
        else:
            rows = []
            for i, s in enumerate(scorers, 1):
                goals  = s.get("goals") or 0
                played = max(s.get("played") or 1, 1)
                rows.append({
                    "#":           i,
                    "Player":      s.get("name",""),
                    "Team":        s.get("team",""),
                    "Nationality": s.get("nationality",""),
                    "Goals":       goals,
                    "Assists":     s.get("assists") or 0,
                    "Penalties":   s.get("penalties") or 0,
                    "Games":       s.get("played") or 0,
                    "Goals/Game":  round(goals / played, 2),
                })
            st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True, height=600)
            st.markdown("#### Goals scored")
            st.bar_chart(pd.DataFrame(rows).head(15).set_index("Player")["Goals"], use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — REMAINING MATCHES
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎯 Remaining Matches — Predictions")
    st.markdown("Win probabilities based on current tournament form.")

    scheduled       = data.get("matches",{}).get("scheduled",[])
    knockout_stages = {"ROUND_OF_16","QUARTER_FINALS","SEMI_FINALS","FINAL","THIRD_PLACE"}
    upcoming        = [m for m in scheduled if m.get("stage","") in knockout_stages]

    if upcoming:
        stage_order_map = {
            "ROUND_OF_16": 1, "QUARTER_FINALS": 2,
            "SEMI_FINALS": 3, "THIRD_PLACE": 4, "FINAL": 5,
        }
        upcoming.sort(key=lambda m: (stage_order_map.get(m.get("stage",""), 9), m.get("date","")))

        current_stage = None
        for match in upcoming:
            home = str(match.get("home") or "").strip()
            away = str(match.get("away") or "").strip()
            if not home or not away or home == "None" or away == "None":
                continue

            stage = match.get("stage","").replace("_"," ").title()
            if stage != current_stage:
                st.markdown(f"### {stage}")
                current_stage = stage

            try:
                ph, pd_, pa = sim.predict_proba(home, away, neutral=True)
            except Exception:
                ph, pd_, pa = 0.4, 0.2, 0.4

            fav = max(ph, pd_, pa)

            with st.container():
                c_info, c_pred, c_watch = st.columns([2, 4, 1])
                with c_info:
                    st.markdown(f"**{home} vs {away}**")
                    st.caption(f"📅 {match.get('date','')}  ·  📍 {match.get('venue','')}")
                with c_pred:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(home, f"{ph*100:.0f}%", "⭐" if ph==fav else "")
                        st.progress(ph)
                    with c2:
                        st.metric("Draw", f"{pd_*100:.0f}%")
                        st.progress(pd_)
                    with c3:
                        st.metric(away, f"{pa*100:.0f}%", "⭐" if pa==fav else "")
                        st.progress(pa)
                with c_watch:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.link_button("📺 Watch", "https://hubafoot.com", use_container_width=True)
            st.divider()

    else:
        st.info("No upcoming knockout matches from API yet — showing hardcoded R16 predictions.")
        hardcoded = [
            {"stage": "Round of 16", "date": "Jul 5",  "home": "Norway",       "away": "Brazil",        "venue": "New York"},
            {"stage": "Round of 16", "date": "Jul 5",  "home": "Portugal",     "away": "Spain",         "venue": "Dallas"},
            {"stage": "Round of 16", "date": "Jul 6",  "home": "England",      "away": "Mexico",        "venue": "Los Angeles"},
            {"stage": "Round of 16", "date": "Jul 7",  "home": "Belgium",      "away": "United States", "venue": "Seattle"},
            {"stage": "Round of 16", "date": "Jul 7",  "home": "Argentina",    "away": "Egypt",         "venue": "Houston"},
            {"stage": "Round of 16", "date": "Jul 8",  "home": "Colombia",     "away": "Switzerland",   "venue": "Miami"},
        ]
        current_stage = None
        for m in hardcoded:
            if m["stage"] != current_stage:
                st.markdown(f"### {m['stage']}")
                current_stage = m["stage"]
            home, away = m["home"], m["away"]
            ph, pd_, pa = sim.predict_proba(home, away, neutral=True)
            fav = max(ph, pd_, pa)
            with st.container():
                c_info, c_pred, c_watch = st.columns([2, 4, 1])
                with c_info:
                    st.markdown(f"**{home} vs {away}**")
                    st.caption(f"📅 {m['date']}  ·  📍 {m['venue']}")
                with c_pred:
                    c1,c2,c3 = st.columns(3)
                    with c1: st.metric(home, f"{ph*100:.0f}%", "⭐" if ph==fav else ""); st.progress(ph)
                    with c2: st.metric("Draw", f"{pd_*100:.0f}%"); st.progress(pd_)
                    with c3: st.metric(away, f"{pa*100:.0f}%", "⭐" if pa==fav else ""); st.progress(pa)
                with c_watch:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.link_button("📺 Watch", "https://hubafoot.com", use_container_width=True)
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
                st.metric(f"{medal} {row['Team']}", f"{row['Win Probability']}%", f"{row['Simulated Wins']:,} wins")
        st.bar_chart(results.head(16).set_index("Team")["Win Probability"], use_container_width=True)
        with st.expander("Full table"):
            st.dataframe(results, use_container_width=True, height=500)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MATCH PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("⚽ Match Predictor")
    all_teams = sorted(TEAM_ELO.keys())
    c1,c2,c3 = st.columns([5,1,5])
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
            home: [TEAM_ELO.get(home,"N/A"), f"{hs.get('form',0):.3f}",
                   f"{hs.get('avg_goals',0):.2f}", f"{hs.get('avg_conceded',0):.2f}",
                   "🟢 live" if hs.get("live") else "📚 historical",
                   hs.get("tournament_games",0)],
            away: [TEAM_ELO.get(away,"N/A"), f"{as_.get('form',0):.3f}",
                   f"{as_.get('avg_goals',0):.2f}", f"{as_.get('avg_conceded',0):.2f}",
                   "🟢 live" if as_.get("live") else "📚 historical",
                   as_.get("tournament_games",0)],
        }).set_index("Stat")
        st.dataframe(comp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — WATCH LIVE (new!)
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📺 Watch World Cup 2026 Live")
    st.markdown("Stream every match for free using the links below.")
    st.divider()

    # Today's / upcoming matches from API
    scheduled       = data.get("matches",{}).get("scheduled",[])
    live_now        = data.get("matches",{}).get("live",[])
    knockout_stages = {"ROUND_OF_16","QUARTER_FINALS","SEMI_FINALS","FINAL","THIRD_PLACE","GROUP_STAGE"}
    all_upcoming    = live_now + [m for m in scheduled if m.get("stage","") in knockout_stages]

    if live_now:
        st.markdown("### 🔴 Live Right Now")
        for m in live_now:
            home_name = m.get("home") or "TBD"
            away_name = m.get("away") or "TBD"
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {home_name} {m['score_home']} — {m['score_away']} {away_name}")
                st.caption(f"📍 {m.get('venue','')}")
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.link_button("📺 Watch Now", "https://hubafoot.com",
                               type="primary", use_container_width=True)
        st.divider()

    st.markdown("### 📅 Upcoming Matches")

    upcoming_display = [m for m in scheduled if m.get("stage","") in
                        {"ROUND_OF_16","QUARTER_FINALS","SEMI_FINALS","FINAL","THIRD_PLACE"}]
    upcoming_display.sort(key=lambda m: m.get("date",""))

    if upcoming_display:
        for m in upcoming_display:
            home_name = str(m.get("home") or "").strip()
            away_name = str(m.get("away") or "").strip()
            if not home_name or not away_name or home_name == "None" or away_name == "None":
                home_name = "TBD"
                away_name = "TBD"

            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.markdown(f"**{home_name} vs {away_name}**")
                st.caption(f"📅 {m.get('date','')}  ·  🕐 {m.get('time','')} UTC  ·  📍 {m.get('venue','')}")
            with c2:
                stage = m.get("stage","").replace("_"," ").title()
                st.caption(f"🏆 {stage}")
            with c3:
                st.link_button("📺 Watch Free", "https://hubafoot.com", use_container_width=True)
            st.divider()
    else:
        # Fallback hardcoded
        hardcoded_watch = [
            {"date": "Jul 5",  "time": "22:00", "home": "Norway",    "away": "Brazil",        "venue": "New York",    "stage": "Round of 16"},
            {"date": "Jul 5",  "time": "21:00", "home": "Portugal",  "away": "Spain",         "venue": "Dallas",      "stage": "Round of 16"},
            {"date": "Jul 6",  "time": "21:00", "home": "England",   "away": "Mexico",        "venue": "Los Angeles", "stage": "Round of 16"},
            {"date": "Jul 7",  "time": "21:00", "home": "Belgium",   "away": "United States", "venue": "Seattle",     "stage": "Round of 16"},
            {"date": "Jul 7",  "time": "21:00", "home": "Argentina", "away": "Egypt",         "venue": "Houston",     "stage": "Round of 16"},
            {"date": "Jul 8",  "time": "21:00", "home": "Colombia",  "away": "Switzerland",   "venue": "Miami",       "stage": "Round of 16"},
            {"date": "Jul 9",  "time": "21:00", "home": "France",    "away": "Morocco",       "venue": "Boston",      "stage": "Quarter-final"},
            {"date": "Jul 19", "time": "19:00", "home": "TBD",       "away": "TBD",           "venue": "MetLife Stadium, New Jersey", "stage": "🏆 Final"},
        ]
        for m in hardcoded_watch:
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.markdown(f"**{m['home']} vs {m['away']}**")
                st.caption(f"📅 {m['date']}  ·  🕐 {m['time']} ET  ·  📍 {m['venue']}")
            with c2:
                st.caption(f"🏆 {m['stage']}")
            with c3:
                st.link_button("📺 Watch Free", "https://hubafoot.com", use_container_width=True)
            st.divider()

    st.divider()

    # Streaming options
    st.markdown("### 🌍 Free Streaming Options")
    st.markdown("Multiple free options to watch every World Cup match:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📺 Huba Football")
        st.markdown("Free streams for all matches. No account needed.")
        st.link_button("Open Huba Football", "https://hubafoot.com",
                       type="primary", use_container_width=True)
    with col2:
        st.markdown("#### 🌐 FIFA+")
        st.markdown("Official FIFA streaming. Free in many countries.")
        st.link_button("Open FIFA+", "https://www.fifa.com/fifaplus",
                       use_container_width=True)
    with col3:
        st.markdown("#### 📡 LiveSoccerTV")
        st.markdown("Find official broadcasters in your country.")
        st.link_button("Find Broadcasters", "https://www.livesoccertv.com",
                       use_container_width=True)

    st.divider()
    st.caption("⚠️ Links open in a new tab. We don't host any streams — we only link to external sites.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — GROUPS & TEAMS
# ════════════════════════════════════════════════════════════════════════════════
with tab6:
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