"""
ScoreCast — Streamlit frontend.
Calls the same prediction logic as the FastAPI backend directly (no HTTP).
"""

import asyncio
import streamlit as st
from datetime import date, timedelta
from src.config import ESPN_FOOTBALL_LEAGUES
from src.predictor import get_all_football_predictions, get_all_basketball_predictions

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoreCast — AI Score Predictions",
    page_icon="⚽",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .card {
    background: #141d2e;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
  }
  .league-tag { font-size: 11px; color: #718096; font-weight: 600; }
  .teams-row { font-size: 22px; font-weight: 800; margin: 4px 0 2px; }
  .score { color: #00d4ff; font-size: 28px; font-weight: 900; }
  .prob-row { font-size: 12px; color: #718096; margin: 6px 0; }
  .prob-w { color: #00e676; font-weight: 700; }
  .prob-d { color: #ffd740; font-weight: 700; }
  .prob-l { color: #ff4444; font-weight: 700; }
  .scoreline-chip {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid #1e2d45;
    border-radius: 7px;
    padding: 3px 9px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 5px;
    margin-bottom: 4px;
  }
  .scoreline-chip.top { border-color: #00d4ff; color: #00d4ff; }
  .safe-bet-box {
    background: rgba(0,230,118,0.08);
    border: 1px solid rgba(0,230,118,0.3);
    border-radius: 9px;
    padding: 10px 14px;
    display: inline-block;
    min-width: 140px;
  }
  .safe-label { font-size: 10px; color: #00e676; font-weight: 700; letter-spacing: 0.5px; }
  .safe-line { font-size: 22px; font-weight: 900; color: #00e676; }
  .safe-prob { font-size: 11px; color: #718096; }
  .form-dot {
    display: inline-block;
    width: 22px; height: 22px;
    border-radius: 50%;
    text-align: center; line-height: 22px;
    font-size: 10px; font-weight: 700;
    margin-right: 3px;
  }
  .form-dot.W { background: #00e676; color: #000; }
  .form-dot.D { background: #ffd740; color: #000; }
  .form-dot.L { background: #ff4444; color: #fff; }
  .h2h-row {
    display: flex; align-items: center; gap: 10px;
    font-size: 12px; padding: 4px 0;
    border-bottom: 1px solid #1e2d45;
  }
  .h2h-date { color: #4a5568; width: 70px; font-size: 11px; }
  .h2h-teams { flex: 1; color: #718096; }
  .h2h-score { font-weight: 800; color: #00d4ff; }
  .h2h-pred { color: #ffd740; font-weight: 700; font-size: 11px; }
  .section-label { font-size: 10px; color: #4a5568; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def form_dot(gf, ga):
    if gf > ga:   return '<span class="form-dot W">W</span>'
    if gf == ga:  return '<span class="form-dot D">D</span>'
    return '<span class="form-dot L">L</span>'

def short_date(iso):
    try:
        return date.fromisoformat(iso[:10]).strftime("%-d %b %y")
    except Exception:
        return "—"

def poisson_mode(lam):
    return max(0, int(lam))

HOME_ADV = 1.1

def render_match_card(m):
    p   = m["prediction"]
    f   = m.get("features", {})
    sb  = p.get("safe_bet")
    lh  = f.get("lambda_home", 1.0)
    la  = f.get("lambda_away", 1.0)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="card">
      <div class="league-tag">⚽ {m.get('league','')} &nbsp;·&nbsp; {m.get('match_time','')[:10]}</div>
      <div class="teams-row">
        {m['home_team']} <span style="color:#4a5568;font-size:18px">vs</span> {m['away_team']}
      </div>
      <div class="score">{p['predicted_home']} – {p['predicted_away']}</div>
      <div class="prob-row">
        <span class="prob-w">W {p['win_probability']}%</span> &nbsp;
        <span class="prob-d">D {p['draw_probability']}%</span> &nbsp;
        <span class="prob-l">L {p['loss_probability']}%</span>
        &nbsp;·&nbsp; 🎯 {p['confidence']}% confidence
      </div>
    """, unsafe_allow_html=True)

    # ── Scorelines + Safe Bet ─────────────────────────────────────────────
    col_s, col_b = st.columns([3, 1])
    with col_s:
        chips = "".join(
            f'<span class="scoreline-chip {"top" if i==0 else ""}">'
            f'{s["scoreline"]} <span style="opacity:.6">{s["probability"]}%</span></span>'
            for i, s in enumerate(p.get("top_scorelines", []))
        )
        st.markdown(f'<div class="section-label">MOST LIKELY SCORELINES</div>{chips}',
                    unsafe_allow_html=True)

    with col_b:
        if sb:
            last_h2h = m.get("h2h", [])
            last_html = ""
            if last_h2h:
                g = last_h2h[0]
                last_html = (f'<div style="font-size:10px;color:#4a5568;margin-top:6px;border-top:1px solid #1e2d45;padding-top:6px">'
                             f'Last meeting<br><strong style="color:#e2e8f0">'
                             f'{g["home_team"]} {g["home_goals"]}–{g["away_goals"]} {g["away_team"]}</strong></div>')
            st.markdown(f"""
            <div class="safe-bet-box">
              <div class="safe-label">SAFE BET</div>
              <div class="safe-line">Over {sb['line']}</div>
              <div class="safe-prob">{sb['probability']}% probability</div>
              {last_html}
            </div>""", unsafe_allow_html=True)

    # ── Form ──────────────────────────────────────────────────────────────
    col_h, col_a = st.columns(2)
    with col_h:
        dots = "".join(form_dot(g["goals_for"], g["goals_ag"]) for g in m.get("home_form", []))
        st.markdown(f'<div class="section-label" style="margin-top:12px">{m["home_team"]} — Last 5</div>{dots}',
                    unsafe_allow_html=True)
    with col_a:
        dots = "".join(form_dot(g["goals_for"], g["goals_ag"]) for g in m.get("away_form", []))
        st.markdown(f'<div class="section-label" style="margin-top:12px">{m["away_team"]} — Last 5</div>{dots}',
                    unsafe_allow_html=True)

    # ── H2H ───────────────────────────────────────────────────────────────
    h2h = m.get("h2h", [])
    if h2h:
        rows = ""
        for g in h2h:
            home_at_home = g.get("fixture_home_at_home", True)
            pred_lh = lh if home_at_home else la * HOME_ADV
            pred_la = la if home_at_home else lh / HOME_ADV
            rows += (f'<div class="h2h-row">'
                     f'<span class="h2h-date">{short_date(g["date"])}</span>'
                     f'<span class="h2h-teams">{g["home_team"]} vs {g["away_team"]}</span>'
                     f'<span class="h2h-score">{g["home_goals"]}–{g["away_goals"]}</span>'
                     f'<span class="h2h-pred">pred {poisson_mode(pred_lh)}–{poisson_mode(pred_la)}</span>'
                     f'</div>')
        st.markdown(f'<div class="section-label" style="margin-top:12px">HEAD TO HEAD — LAST 5</div>{rows}',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_basketball_card(m):
    p = m["prediction"]
    st.markdown(f"""
    <div class="card">
      <div class="league-tag">🏀 NBA &nbsp;·&nbsp; {m.get('status','')}</div>
      <div class="teams-row">
        {m['home_team']} <span style="color:#4a5568;font-size:18px">vs</span> {m['away_team']}
      </div>
      <div class="score" style="color:#ff9100">{p.get('predicted_home','—')} – {p.get('predicted_away','—')}</div>
      <div class="prob-row">
        <span class="prob-w">Home {p.get('home_win_prob',0)}%</span> &nbsp;
        <span class="prob-l">Away {p.get('away_win_prob',0)}%</span>
        &nbsp;·&nbsp; 🎯 {p.get('confidence','—')}% confidence
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽🏀 ScoreCast")
    sport = st.radio("Sport", ["Football", "Basketball"], horizontal=True)

    selected_date = st.date_input("Date", value=date.today(),
                                  min_value=date.today() - timedelta(days=1),
                                  max_value=date.today() + timedelta(days=7))

    if sport == "Football":
        league = st.selectbox("League", list(ESPN_FOOTBALL_LEAGUES.keys()))

    if st.button("🔄 Load Predictions", use_container_width=True):
        st.session_state.pop("results", None)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("ScoreCast — AI Score Predictions")

if sport == "Football":
    if "results" not in st.session_state or st.session_state.get("last_query") != (sport, str(selected_date), league):
        with st.spinner(f"Fetching {league} fixtures for {selected_date}…"):
            try:
                results = asyncio.run(
                    get_all_football_predictions(league, target_date=str(selected_date))
                )
                st.session_state["results"]    = results
                st.session_state["last_query"] = (sport, str(selected_date), league)
            except Exception as e:
                st.error(f"Error loading predictions: {e}")
                results = []
    else:
        results = st.session_state.get("results", [])

    if not results:
        st.info(f"No {league} fixtures found for {selected_date}.")
    else:
        # Optional team filter
        teams = sorted({m["home_team"] for m in results} | {m["away_team"] for m in results})
        team_filter = st.selectbox("Filter by team", ["All Teams"] + teams)
        if team_filter != "All Teams":
            results = [m for m in results
                       if m["home_team"] == team_filter or m["away_team"] == team_filter]
        st.caption(f"{len(results)} match{'es' if len(results)!=1 else ''}")
        for m in results:
            render_match_card(m)

else:  # Basketball
    if "results" not in st.session_state or st.session_state.get("last_query") != (sport,):
        with st.spinner("Fetching today's NBA games…"):
            try:
                results = get_all_basketball_predictions()
                st.session_state["results"]    = results
                st.session_state["last_query"] = (sport,)
            except Exception as e:
                st.error(f"Error loading predictions: {e}")
                results = []
    else:
        results = st.session_state.get("results", [])

    if not results:
        st.info("No NBA games scheduled today.")
    else:
        for m in results:
            render_basketball_card(m)
