"""
Data fetcher — pulls fixtures, results, and stats from external APIs.
Football: ESPN public API (free, no key, current season data)
Basketball: nba_api (free, no key needed)
"""

import httpx
import asyncio
from datetime import date, timedelta
from typing import Optional
from src.config import API_FOOTBALL_KEY, FOOTBALL_LEAGUES

FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}

ESPN_SOCCER_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


# ─── Football (ESPN) ──────────────────────────────────────────────────────────

async def get_espn_soccer_fixtures(league_slug: str, target_date: Optional[str] = None) -> list:
    """Fetch today's soccer fixtures from ESPN for a given league."""
    if not target_date:
        target_date = date.today().isoformat()
    date_param = target_date.replace("-", "")  # ESPN expects YYYYMMDD

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ESPN_SOCCER_BASE}/{league_slug}/scoreboard",
            params={"dates": date_param},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

    league_name = data.get("leagues", [{}])[0].get("name", "")
    fixtures = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        competitors = comp["competitors"]
        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue
        fixtures.append({
            "fixture_id":   event["id"],
            "date":         event["date"],
            "status":       comp.get("status", {}).get("type", {}).get("name", ""),
            "home_team":    home["team"]["displayName"],
            "home_team_id": home["team"]["id"],
            "away_team":    away["team"]["displayName"],
            "away_team_id": away["team"]["id"],
            "venue":        comp.get("venue", {}).get("fullName", ""),
            "league":       league_name,
            "league_slug":  league_slug,
        })
    return fixtures


async def get_espn_team_schedule_raw(team_id: str, league_slug: str) -> list:
    """Return all ESPN events for a team this season (raw event dicts)."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ESPN_SOCCER_BASE}/{league_slug}/teams/{team_id}/schedule",
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    return data.get("events", [])


async def get_espn_team_match_history(team_id: str, league_slug: str, n: int = 10) -> list:
    """Fetch last N completed matches for a team from ESPN."""
    events = await get_espn_team_schedule_raw(team_id, league_slug)

    matches = []
    for event in events:
        comp = event["competitions"][0]
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue
        competitors = comp["competitors"]
        our = next((c for c in competitors if c["id"] == str(team_id)), None)
        opp = next((c for c in competitors if c["id"] != str(team_id)), None)
        if not our or not opp:
            continue
        our_score = our.get("score", {})
        opp_score = opp.get("score", {})
        goals_for = our_score.get("value") if isinstance(our_score, dict) else our_score
        goals_ag  = opp_score.get("value") if isinstance(opp_score, dict) else opp_score
        if goals_for is None or goals_ag is None:
            continue
        matches.append({
            "fixture_id":  event["id"],
            "date":        event["date"],
            "is_home":     our["homeAway"] == "home",
            "goals_for":   int(goals_for),
            "goals_ag":    int(goals_ag),
            "opponent_id": opp["id"],
        })

    matches.sort(key=lambda x: x["date"], reverse=True)
    return matches[:n]


async def get_espn_head_to_head(home_id: str, away_id: str, league_slug: str, last: int = 10) -> list:
    """Find H2H results by cross-referencing both teams' schedules."""
    home_events, away_events = await asyncio.gather(
        get_espn_team_schedule_raw(home_id, league_slug),
        get_espn_team_schedule_raw(away_id, league_slug),
    )
    away_ids = {e["id"] for e in away_events}
    h2h = []
    for event in home_events:
        if event["id"] not in away_ids:
            continue
        comp = event["competitions"][0]
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue
        competitors = comp["competitors"]
        home_c = next((c for c in competitors if c["id"] == str(home_id)), None)
        away_c = next((c for c in competitors if c["id"] == str(away_id)), None)
        if not home_c or not away_c:
            continue
        hs = home_c.get("score", {})
        as_ = away_c.get("score", {})
        hg = hs.get("value") if isinstance(hs, dict) else hs
        ag = as_.get("value") if isinstance(as_, dict) else as_
        if hg is None or ag is None:
            continue
        h2h.append({
            "date":       event["date"],
            "home_team":  home_c["team"]["displayName"],
            "away_team":  away_c["team"]["displayName"],
            "home_goals": int(hg),
            "away_goals": int(ag),
        })

    h2h.sort(key=lambda x: x["date"], reverse=True)
    return h2h[:last]


# ─── Football (API-Football — legacy/fallback) ────────────────────────────────

async def get_football_fixtures(league_id: int, season: int, target_date: Optional[str] = None) -> list:
    """Fetch fixtures for a league on a given date (default: today)."""
    if not target_date:
        target_date = date.today().isoformat()

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{FOOTBALL_BASE}/fixtures",
            headers=HEADERS,
            params={"league": league_id, "season": season, "date": target_date},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

    fixtures = []
    for f in data.get("response", []):
        fixtures.append({
            "fixture_id":    f["fixture"]["id"],
            "date":          f["fixture"]["date"],
            "status":        f["fixture"]["status"]["short"],
            "home_team":     f["teams"]["home"]["name"],
            "home_team_id":  f["teams"]["home"]["id"],
            "away_team":     f["teams"]["away"]["name"],
            "away_team_id":  f["teams"]["away"]["id"],
            "venue":         f["fixture"]["venue"]["name"],
            "league":        f["league"]["name"],
            "season":        f["league"]["season"],
        })
    return fixtures


async def get_team_last_n_matches(team_id: int, season: int, n: int = 10) -> list:
    """Fetch last N completed matches for a team."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{FOOTBALL_BASE}/fixtures",
            headers=HEADERS,
            params={"team": team_id, "season": season, "last": n, "status": "FT"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

    matches = []
    for f in data.get("response", []):
        home_id    = f["teams"]["home"]["id"]
        is_home    = home_id == team_id
        goals_for  = f["goals"]["home"] if is_home else f["goals"]["away"]
        goals_ag   = f["goals"]["away"] if is_home else f["goals"]["home"]
        matches.append({
            "fixture_id":  f["fixture"]["id"],
            "date":        f["fixture"]["date"],
            "is_home":     is_home,
            "goals_for":   goals_for,
            "goals_ag":    goals_ag,
            "opponent_id": f["teams"]["away"]["id"] if is_home else home_id,
        })
    return matches


async def get_head_to_head_football(team1_id: int, team2_id: int, last: int = 10) -> list:
    """Fetch head-to-head results between two teams."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{FOOTBALL_BASE}/fixtures/headtohead",
            headers=HEADERS,
            params={"h2h": f"{team1_id}-{team2_id}", "last": last, "status": "FT"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

    return [
        {
            "date":       f["fixture"]["date"],
            "home_team":  f["teams"]["home"]["name"],
            "away_team":  f["teams"]["away"]["name"],
            "home_goals": f["goals"]["home"],
            "away_goals": f["goals"]["away"],
        }
        for f in data.get("response", [])
    ]


async def get_team_injuries_football(team_id: int, fixture_id: int) -> list:
    """Check injury/suspension list for a team going into a fixture."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{FOOTBALL_BASE}/injuries",
            headers=HEADERS,
            params={"team": team_id, "fixture": fixture_id},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

    return [
        {
            "player": p["player"]["name"],
            "reason": p["player"]["reason"],
            "type":   p["player"]["type"],
        }
        for p in data.get("response", [])
    ]


# ─── Basketball (NBA) ─────────────────────────────────────────────────────────

def get_nba_today_scoreboard() -> list:
    """Get today's NBA games."""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        sb   = scoreboard.ScoreBoard()
        data = sb.get_dict()
        games = []
        for g in data.get("scoreboard", {}).get("games", []):
            games.append({
                "game_id":    g["gameId"],
                "status":     g["gameStatusText"],
                "home_team":  g["homeTeam"]["teamName"],
                "home_abbr":  g["homeTeam"]["teamTricode"],
                "away_team":  g["awayTeam"]["teamName"],
                "away_abbr":  g["awayTeam"]["teamTricode"],
                "home_score": g["homeTeam"].get("score", 0),
                "away_score": g["awayTeam"].get("score", 0),
            })
        return games
    except Exception as e:
        print(f"[NBA scoreboard error] {e}")
        return []


def get_nba_team_last_n_games(team_id: int, n: int = 10) -> list:
    """Fetch last N game logs for an NBA team."""
    try:
        from nba_api.stats.endpoints import teamgamelogs
        from nba_api.stats.static import teams as nba_teams
        import time
        time.sleep(0.6)   # rate limit courtesy pause

        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=str(team_id),
            last_n_games_nullable=n,
        ).get_data_frames()[0]

        results = []
        for _, row in logs.iterrows():
            results.append({
                "game_id":     row["GAME_ID"],
                "date":        row["GAME_DATE"],
                "is_home":     "@" not in str(row.get("MATCHUP", "")),
                "pts_for":     row["PTS"],
                "pts_ag":      row["PTS"] - row["PLUS_MINUS"],
                "fg_pct":      row.get("FG_PCT", 0),
                "fg3_pct":     row.get("FG3_PCT", 0),
                "ft_pct":      row.get("FT_PCT", 0),
                "reb":         row.get("REB", 0),
                "ast":         row.get("AST", 0),
                "tov":         row.get("TOV", 0),
            })
        return results
    except Exception as e:
        print(f"[NBA game logs error] {e}")
        return []


def get_nba_all_teams() -> list:
    """Return all NBA team id/name mappings."""
    from nba_api.stats.static import teams as nba_teams
    return nba_teams.get_teams()
