"""
Data fetcher — pulls fixtures, results, and stats from external APIs.
Football: ESPN public API (free, no key, current season data)
Basketball: nba_api (free, no key needed)
"""

import httpx
import asyncio
from datetime import date, timedelta
from typing import Optional
from src.config import API_FOOTBALL_KEY, FOOTBALL_LEAGUES, FOOTBALL_DATA_KEY, THESPORTSDB_KEY

FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}

ESPN_SOCCER_BASE   = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_STANDINGS_BASE = "https://site.api.espn.com/apis/v2/sports/soccer"


# ─── Football (ESPN) ──────────────────────────────────────────────────────────

async def get_espn_standings(league_slug: str) -> dict:
    """
    Fetch current league standings from ESPN.
    Returns dict keyed by team_id → {rank, points, wins, draws, losses, games_played,
                                      goals_for, goals_against, goal_diff}
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ESPN_STANDINGS_BASE}/{league_slug}/standings",
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

    standings = {}
    try:
        entries = data["children"][0]["standings"]["entries"]
        for entry in entries:
            stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
            team  = entry.get("team", {})
            team_id = str(team.get("id", "")) if isinstance(team, dict) else ""
            if not team_id:
                continue
            standings[team_id] = {
                "rank":         int(stats.get("rank", 99)),
                "points":       int(stats.get("points", 0)),
                "wins":         int(stats.get("wins", 0)),
                "draws":        int(stats.get("ties", 0)),
                "losses":       int(stats.get("losses", 0)),
                "games_played": int(stats.get("gamesPlayed", 0)),
            }
    except (KeyError, IndexError, TypeError):
        pass
    return standings


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
        def _logo(team: dict) -> str:
            logos = team.get("logos") or []
            if logos:
                return logos[0].get("href", "")
            # fallback: ESPN CDN pattern
            tid = team.get("id", "")
            return f"https://a.espncdn.com/i/teamlogos/soccer/500/{tid}.png" if tid else ""

        status_obj  = comp.get("status", {})
        status_type = status_obj.get("type", {})
        status_name = status_type.get("name", "")
        is_final    = status_type.get("completed", False)
        is_live     = not is_final and status_name in (
            "STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_FIRST_HALF",
            "STATUS_SECOND_HALF", "STATUS_EXTRA_TIME", "STATUS_PENALTY",
        )
        display_clock = status_obj.get("displayClock") or status_type.get("shortDetail", "")

        def _score(competitor):
            s = competitor.get("score", None)
            if s is None:
                return None
            if isinstance(s, dict):
                v = s.get("value")
                return int(v) if v is not None else None
            try:
                return int(s)
            except (TypeError, ValueError):
                return None

        def _linescore(competitor, period_idx):
            ls = competitor.get("linescores") or []
            if period_idx < len(ls):
                v = ls[period_idx].get("value")
                try:
                    return int(v) if v is not None else None
                except (TypeError, ValueError):
                    return None
            return None

        # HT available when in 2nd half, extra time, or completed
        ht_available = is_final or status_name in (
            "STATUS_SECOND_HALF", "STATUS_HALFTIME",
            "STATUS_EXTRA_TIME", "STATUS_PENALTY",
        )

        fixtures.append({
            "fixture_id":    event["id"],
            "date":          event["date"],
            "status":        display_clock or status_name,
            "is_live":       is_live,
            "is_final":      is_final,
            "home_team":     home["team"]["displayName"],
            "home_team_id":  home["team"]["id"],
            "home_team_logo": _logo(home["team"]),
            "home_goals":    _score(home) if (is_live or is_final) else None,
            "home_goals_ht": _linescore(home, 0) if ht_available else None,
            "away_team":     away["team"]["displayName"],
            "away_team_id":  away["team"]["id"],
            "away_team_logo": _logo(away["team"]),
            "away_goals":    _score(away) if (is_live or is_final) else None,
            "away_goals_ht": _linescore(away, 0) if ht_available else None,
            "venue":         comp.get("venue", {}).get("fullName", ""),
            "league":        league_name,
            "league_slug":   league_slug,
        })
    return fixtures


async def get_espn_team_schedule_raw(team_id: str, league_slug: str, season: int = None) -> list:
    """Return all ESPN events for a team in a given season (raw event dicts)."""
    params = {}
    if season:
        params["season"] = season
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ESPN_SOCCER_BASE}/{league_slug}/teams/{team_id}/schedule",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    return data.get("events", [])


def _parse_events(events: list, team_id: str, comp_tag: str = "league") -> list:
    """Parse ESPN event list into match dicts for a given team."""
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
            "comp":        comp_tag,
        })
    return matches


async def get_espn_team_match_history(team_id: str, league_slug: str, n: int = 10) -> list:
    """Fetch last N completed league matches for a team from ESPN."""
    events = await get_espn_team_schedule_raw(team_id, league_slug)
    matches = _parse_events(events, team_id, comp_tag="league")
    matches.sort(key=lambda x: x["date"], reverse=True)
    return matches[:n]


async def get_espn_team_all_matches(team_id: str, league_slug: str, n: int = 20) -> list:
    """
    Fetch completed matches across ALL competitions (league + cups).
    Used for rest/congestion calculations so we know the true last match date.
    """
    from src.config import ESPN_CUP_SLUGS
    cup_slugs = ESPN_CUP_SLUGS.get(league_slug, [])

    tasks = [get_espn_team_schedule_raw(team_id, league_slug)]
    for slug in cup_slugs:
        tasks.append(get_espn_team_schedule_raw(team_id, slug))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_matches = []
    tags = ["league"] + cup_slugs
    for tag, result in zip(tags, results):
        if isinstance(result, Exception):
            continue
        all_matches.extend(_parse_events(result, team_id, comp_tag=tag))

    # Deduplicate by fixture_id (some matches might appear in multiple feeds)
    seen = set()
    unique = []
    for m in all_matches:
        if m["fixture_id"] not in seen:
            seen.add(m["fixture_id"])
            unique.append(m)

    unique.sort(key=lambda x: x["date"], reverse=True)
    return unique[:n]


async def get_intl_team_all_matches(team_id: str, n: int = 20) -> list:
    """
    Fetch completed matches for a national team across ALL international competitions.
    Checks World Cup, qualifiers, Nations League, friendlies.
    """
    from src.config import INTERNATIONAL_COMP_SLUGS

    tasks = [get_espn_team_schedule_raw(team_id, slug) for slug in INTERNATIONAL_COMP_SLUGS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_matches = []
    for slug, result in zip(INTERNATIONAL_COMP_SLUGS, results):
        if isinstance(result, Exception):
            continue
        all_matches.extend(_parse_events(result, team_id, comp_tag=slug))

    seen = set()
    unique = []
    for m in all_matches:
        if m["fixture_id"] not in seen:
            seen.add(m["fixture_id"])
            unique.append(m)

    unique.sort(key=lambda x: x["date"], reverse=True)
    return unique[:n]


async def get_intl_head_to_head(home_id: str, away_id: str, last: int = 5) -> list:
    """Find H2H results between two national teams across all international competitions."""
    from src.config import INTERNATIONAL_COMP_SLUGS

    home_tasks = [get_espn_team_schedule_raw(home_id, slug) for slug in INTERNATIONAL_COMP_SLUGS]
    away_tasks = [get_espn_team_schedule_raw(away_id, slug) for slug in INTERNATIONAL_COMP_SLUGS]
    all_results = await asyncio.gather(*(home_tasks + away_tasks), return_exceptions=True)

    home_results = all_results[:len(INTERNATIONAL_COMP_SLUGS)]
    away_results = all_results[len(INTERNATIONAL_COMP_SLUGS):]

    away_event_ids = set()
    for result in away_results:
        if isinstance(result, Exception):
            continue
        for event in result:
            away_event_ids.add(event["id"])

    h2h = []
    for result in home_results:
        if isinstance(result, Exception):
            continue
        for event in result:
            if event["id"] not in away_event_ids:
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
                "home_team": home_c["team"]["displayName"],
                "away_team": away_c["team"]["displayName"],
                "home_goals": int(hg),
                "away_goals": int(ag),
                "date": event["date"],
                "fixture_home_at_home": home_c["homeAway"] == "home",
            })
            if len(h2h) >= last:
                break
        if len(h2h) >= last:
            break

    h2h.sort(key=lambda x: x["date"], reverse=True)
    return h2h[:last]


async def get_espn_head_to_head(home_id: str, away_id: str, league_slug: str, last: int = 5) -> list:
    """
    Find H2H results across multiple seasons until we have `last` meetings.
    Checks current season first, then walks back year by year (up to 4 seasons).
    """
    from datetime import date as _date
    current_season = _date.today().year  # ESPN season = year the season started

    h2h: list = []
    seasons_checked = 0

    for offset in range(4):  # check up to 4 seasons back
        season = current_season - offset
        home_events, away_events = await asyncio.gather(
            get_espn_team_schedule_raw(home_id, league_slug, season=season),
            get_espn_team_schedule_raw(away_id, league_slug, season=season),
        )
        away_ids = {e["id"] for e in away_events}
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
                "date":            event["date"],
                "home_team":       home_c["team"]["displayName"],
                "away_team":       away_c["team"]["displayName"],
                "home_goals":      int(hg),
                "away_goals":      int(ag),
                # Was the fixture's home team actually playing at home that day?
                "fixture_home_at_home": home_c.get("homeAway") == "home",
            })
        seasons_checked += 1
        if len(h2h) >= last:
            break

    h2h.sort(key=lambda x: x["date"], reverse=True)
    return h2h[:last]


async def get_espn_fixture_dates_for_month(league_slug: str, year: int, month: int) -> list:
    """
    Return a list of ISO date strings (YYYY-MM-DD) that have fixtures
    for the given league in a given month, using ESPN's scoreboard date-range query.
    """
    import calendar as _cal
    last_day = _cal.monthrange(year, month)[1]
    start = f"{year}{month:02d}01"
    end   = f"{year}{month:02d}{last_day:02d}"

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{ESPN_SOCCER_BASE}/{league_slug}/scoreboard",
            params={"dates": f"{start}-{end}"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

    dates = set()
    for event in data.get("events", []):
        raw = event.get("date", "")
        if raw:
            dates.add(raw[:10])
    return sorted(dates)


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


# ─── Basketball (NBA via ESPN) ────────────────────────────────────────────────

ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# Cache ESPN team id → name mapping
_espn_nba_teams: dict = {}


async def _get_espn_nba_teams() -> dict:
    """Return {team_name_lower: espn_team_id} mapping."""
    global _espn_nba_teams
    if _espn_nba_teams:
        return _espn_nba_teams
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ESPN_NBA_BASE}/teams", params={"limit": 40}, timeout=15)
        r.raise_for_status()
        data = r.json()
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for team in league.get("teams", []):
                t = team.get("team", {})
                tid  = t.get("id", "")
                name = t.get("displayName", "")
                nick = t.get("name", "")          # e.g. "Lakers"
                abbr = t.get("abbreviation", "")
                for key in [name.lower(), nick.lower(), abbr.lower()]:
                    if key:
                        _espn_nba_teams[key] = tid
    return _espn_nba_teams


async def get_espn_nba_scoreboard(date_str: Optional[str] = None) -> list:
    """Get NBA games from ESPN for a given date (ISO YYYY-MM-DD), or today if not provided."""
    params = {}
    if date_str:
        params["dates"] = date_str.replace("-", "")  # ESPN expects YYYYMMDD
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ESPN_NBA_BASE}/scoreboard", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

    games = []
    for event in data.get("events", []):
        comp        = event["competitions"][0]
        competitors = comp["competitors"]
        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue
        status_obj  = comp.get("status", {})
        status_type = status_obj.get("type", {})
        is_final    = status_type.get("completed", False)
        is_live     = not is_final and status_type.get("state", "") == "in"
        display     = status_obj.get("displayClock") or status_type.get("shortDetail", "")
        games.append({
            "game_id":    event["id"],
            "status":     display,
            "is_live":    is_live,
            "is_final":   is_final,
            "home_team":  home["team"]["displayName"],
            "home_abbr":  home["team"].get("abbreviation", ""),
            "home_team_id": home["team"]["id"],
            "away_team":  away["team"]["displayName"],
            "away_abbr":  away["team"].get("abbreviation", ""),
            "away_team_id": away["team"]["id"],
            "home_score": int(home.get("score", 0) or 0),
            "away_score": int(away.get("score", 0) or 0),
            "home_team_logo": (home["team"].get("logos") or [{}])[0].get("href", "")
                              or f"https://a.espncdn.com/i/teamlogos/nba/500/{home['team']['id']}.png",
            "away_team_logo": (away["team"].get("logos") or [{}])[0].get("href", "")
                              or f"https://a.espncdn.com/i/teamlogos/nba/500/{away['team']['id']}.png",
        })
    return games


async def get_espn_nba_dates_for_month(year: int, month: int) -> list:
    """
    Return list of ISO date strings (YYYY-MM-DD) that have at least one NBA event
    in the given month. Uses a single date-range query (fast path), falls back to
    day-by-day with a shared client and concurrency limit.
    """
    import calendar as _cal
    last_day = _cal.monthrange(year, month)[1]
    start = f"{year}{month:02d}01"
    end   = f"{year}{month:02d}{last_day:02d}"

    # Fast path: single range query (same as football)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ESPN_NBA_BASE}/scoreboard",
                params={"dates": f"{start}-{end}"},
                timeout=20,
            )
            if r.status_code == 200:
                dates = set()
                for event in r.json().get("events", []):
                    raw = event.get("date", "")
                    if raw:
                        dates.add(raw[:10])
                if dates:
                    return sorted(dates)
    except Exception:
        pass

    # Fallback: check each day with shared client + semaphore (max 5 concurrent)
    sem = asyncio.Semaphore(5)

    async def _check_day(client: httpx.AsyncClient, day: int) -> Optional[str]:
        iso = f"{year}-{month:02d}-{day:02d}"
        async with sem:
            try:
                r = await client.get(
                    f"{ESPN_NBA_BASE}/scoreboard",
                    params={"dates": f"{year}{month:02d}{day:02d}"},
                    timeout=8,
                )
                if r.status_code == 200 and r.json().get("events"):
                    return iso
            except Exception:
                pass
        return None

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_check_day(client, d) for d in range(1, last_day + 1)])
    return sorted(r for r in results if r is not None)


async def get_espn_nba_full_team_schedule(team_id: str) -> list:
    """Return all events for an NBA team (completed and upcoming) from ESPN."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ESPN_NBA_BASE}/teams/{team_id}/schedule",
                timeout=15,
            )
            if r.status_code != 200:
                return []
            data = r.json()
        return data.get("events", [])
    except Exception:
        return []


async def get_espn_nba_team_games(team_id: str, n: int = 10) -> list:
    """Fetch last N completed games for an NBA team from ESPN."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ESPN_NBA_BASE}/teams/{team_id}/schedule",
                timeout=15,
            )
            if r.status_code != 200:
                return []
            data = r.json()
    except Exception:
        return []

    matches = []
    tid = str(team_id)
    for event in data.get("events", []):
        comp = event["competitions"][0]
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue
        competitors = comp["competitors"]
        # Match by competitor id OR nested team id
        our = next(
            (c for c in competitors
             if c.get("id") == tid or c.get("team", {}).get("id") == tid),
            None,
        )
        opp = next(
            (c for c in competitors
             if c.get("id") != tid and c.get("team", {}).get("id") != tid),
            None,
        )
        if not our or not opp:
            continue
        def _pts(c):
            s = c.get("score", 0)
            return int(s.get("value", 0) if isinstance(s, dict) else (s or 0))

        our_score = _pts(our)
        opp_score = _pts(opp)
        matches.append({
            "game_id":  event["id"],
            "date":     event["date"],
            "is_home":  our["homeAway"] == "home",
            "pts_for":  our_score,
            "pts_ag":   opp_score,
            "fg_pct":   0.46,   # league avg defaults — ESPN schedule doesn't include box stats
            "fg3_pct":  0.36,
            "ft_pct":   0.77,
            "reb":      44,
            "ast":      25,
            "tov":      14,
        })

    matches.sort(key=lambda x: x["date"], reverse=True)
    return matches[:n]


# ── nba_api: team name → nba_api integer ID ───────────────────────────────────

_nba_api_id_map: dict = {}


def _nba_api_team_id(team_name: str) -> Optional[int]:
    """Map an ESPN display name to the nba_api integer team ID."""
    global _nba_api_id_map
    if not _nba_api_id_map:
        try:
            from nba_api.stats.static import teams as _s
            for t in _s.get_teams():
                for key in (t["full_name"].lower(), t["nickname"].lower(), t["abbreviation"].lower()):
                    _nba_api_id_map[key] = t["id"]
        except Exception:
            return None
    name_lower = team_name.lower()
    if name_lower in _nba_api_id_map:
        return _nba_api_id_map[name_lower]
    for key, tid in _nba_api_id_map.items():
        if key and (key in name_lower or name_lower in key):
            return tid
    return None


def _nba_api_fetch_team_games(team_name: str, n: int) -> list:
    """
    Sync: fetch last N game logs from stats.nba.com (real box score stats).
    Returns pts, fg%, 3pt%, ft%, reb, ast, tov per game.
    """
    import time
    try:
        from nba_api.stats.endpoints import teamgamelogs
        nba_id = _nba_api_team_id(team_name)
        if nba_id is None:
            return []
        time.sleep(0.5)
        df = teamgamelogs.TeamGameLogs(
            team_id_nullable=str(nba_id),
            last_n_games_nullable=n,
        ).get_data_frames()[0]
        results = []
        for _, row in df.iterrows():
            results.append({
                "game_id":  str(row["GAME_ID"]),
                "date":     str(row["GAME_DATE"]),
                "is_home":  "@" not in str(row.get("MATCHUP", "")),
                "pts_for":  int(row["PTS"]),
                "pts_ag":   int(row["PTS"]) - int(row["PLUS_MINUS"]),
                "fg_pct":   float(row["FG_PCT"] or 0.46),
                "fg3_pct":  float(row["FG3_PCT"] or 0.36),
                "ft_pct":   float(row["FT_PCT"] or 0.77),
                "reb":      int(row["REB"] or 44),
                "ast":      int(row["AST"] or 25),
                "tov":      int(row["TOV"] or 14),
            })
        return results
    except Exception as e:
        print(f"[nba_api games] {team_name}: {e}")
        return []


async def get_espn_nba_team_stats(team_id: str) -> dict:
    """
    Fetch season-average stats for an NBA team from ESPN.
    Returns {fg_pct, fg3_pct, ft_pct, avg_reb, avg_ast, avg_tov, avg_pts, avg_pts_allowed}.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{ESPN_NBA_BASE}/teams/{team_id}/statistics",
                timeout=10,
            )
            if r.status_code != 200:
                return {}
            data = r.json()
        cats = data.get("results", {}).get("stats", {}).get("categories", [])
        stats: dict = {}
        for cat in cats:
            for s in cat.get("stats", []):
                stats[s["name"]] = s.get("value", 0)
        return {
            "fg_pct":   (stats.get("fieldGoalPct", 46.0)) / 100,
            "fg3_pct":  (stats.get("threePointFieldGoalPct", 36.0)) / 100,
            "ft_pct":   (stats.get("freeThrowPct", 77.0)) / 100,
            "avg_reb":  stats.get("avgRebounds", 44.0),
            "avg_ast":  stats.get("avgAssists", 25.0),
            "avg_tov":  stats.get("avgTurnovers", 14.0),
            "avg_pts":  stats.get("avgPoints", 113.0),
        }
    except Exception:
        return {}


async def get_nba_team_history(espn_team_id: str, team_name: str, n: int = 10) -> list:
    """
    Get team game history enriched with real shooting/rebounding stats.
    - ESPN schedule for per-game scores + dates
    - ESPN team statistics for real fg%, reb, ast, tov (replaces hardcoded defaults)
    - Falls back to nba_api stats if ESPN schedule returns nothing
    """
    games, team_stats = await asyncio.gather(
        get_espn_nba_team_games(espn_team_id, n),
        get_espn_nba_team_stats(espn_team_id),
    )

    if not games:
        print(f"[NBA] ESPN returned 0 games for {team_name}, trying nba_api...")
        return await asyncio.to_thread(_nba_api_fetch_team_games, team_name, n)

    if team_stats:
        for g in games:
            g["fg_pct"]  = team_stats.get("fg_pct",  g["fg_pct"])
            g["fg3_pct"] = team_stats.get("fg3_pct", g["fg3_pct"])
            g["ft_pct"]  = team_stats.get("ft_pct",  g["ft_pct"])
            g["reb"]     = team_stats.get("avg_reb",  g["reb"])
            g["ast"]     = team_stats.get("avg_ast",  g["ast"])
            g["tov"]     = team_stats.get("avg_tov",  g["tov"])

    return games


def _nba_api_live_scoreboard() -> list:
    """Sync: fetch today's games from the NBA live data CDN via nba_api."""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        sb   = scoreboard.ScoreBoard()
        data = sb.get_dict()
        games = []
        for g in data.get("scoreboard", {}).get("games", []):
            is_final = g.get("gameStatus") == 3
            is_live  = g.get("gameStatus") == 2
            games.append({
                "game_id":       g["gameId"],
                "status":        g.get("gameStatusText", ""),
                "is_live":       is_live,
                "is_final":      is_final,
                "home_team":     g["homeTeam"]["teamCity"] + " " + g["homeTeam"]["teamName"],
                "home_abbr":     g["homeTeam"]["teamTricode"],
                "home_team_id":  "",
                "away_team":     g["awayTeam"]["teamCity"] + " " + g["awayTeam"]["teamName"],
                "away_abbr":     g["awayTeam"]["teamTricode"],
                "away_team_id":  "",
                "home_score":    int(g["homeTeam"].get("score") or 0),
                "away_score":    int(g["awayTeam"].get("score") or 0),
                "home_team_logo": "",
                "away_team_logo": "",
            })
        return games
    except Exception as e:
        print(f"[nba_api live] {e}")
        return []


async def get_nba_scoreboard(date_str: Optional[str] = None) -> list:
    """
    NBA games for a given date (or today). ESPN is primary (team IDs + logos).
    Falls back to nba_api live CDN when ESPN fails (today only).
    """
    try:
        games = await get_espn_nba_scoreboard(date_str)
        if games:
            return games
    except Exception as e:
        print(f"[ESPN NBA scoreboard] {e}")

    print("[NBA] ESPN scoreboard failed, trying nba_api live...")
    games = await asyncio.to_thread(_nba_api_live_scoreboard)
    if not games:
        return []

    # Enrich with ESPN team IDs so team history lookups work
    try:
        espn_map = await _get_espn_nba_teams()  # {abbr/nick lower → espn_id}
        for g in games:
            for key in (g["home_abbr"].lower(), g["home_team"].lower().split()[-1]):
                if key in espn_map:
                    g["home_team_id"] = espn_map[key]
                    break
            for key in (g["away_abbr"].lower(), g["away_team"].lower().split()[-1]):
                if key in espn_map:
                    g["away_team_id"] = espn_map[key]
                    break
        # Set logos using ESPN CDN now that we have team IDs
        for g in games:
            if g.get("home_team_id") and not g.get("home_team_logo"):
                g["home_team_logo"] = f"https://a.espncdn.com/i/teamlogos/nba/500/{g['home_team_id']}.png"
            if g.get("away_team_id") and not g.get("away_team_logo"):
                g["away_team_logo"] = f"https://a.espncdn.com/i/teamlogos/nba/500/{g['away_team_id']}.png"
    except Exception:
        pass

    return games


def get_nba_all_teams() -> list:
    """Return all NBA team id/name mappings."""
    from nba_api.stats.static import teams as nba_teams
    return nba_teams.get_teams()


# ─── football-data.org  (HT scores) ──────────────────────────────────────────

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"


def _normalize_team(name: str) -> str:
    """Normalize team name for fuzzy matching across APIs."""
    import re
    n = name.lower()
    # Strip common suffixes/prefixes
    n = re.sub(r'\b(fc|cf|sc|ac|afc|bfc|sfc|fk|sk|if|bk|ik|rsc|vfb|vfl|fsv|rb|as|ss|us|ud|cd|sd|ca|cf|rc|real|club|de|sporting|atletico)\b', '', n)
    n = re.sub(r'[^a-z0-9 ]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def _teams_match(a: str, b: str) -> bool:
    """Fuzzy match two team names from different APIs."""
    na, nb = _normalize_team(a), _normalize_team(b)
    if na == nb:
        return True
    # Check if one contains the first meaningful token of the other
    ta = [t for t in na.split() if len(t) > 2]
    tb = [t for t in nb.split() if len(t) > 2]
    if not ta or not tb:
        return False
    # All tokens of the shorter name present in the longer
    shorter, longer = (ta, nb) if len(ta) <= len(tb) else (tb, na)
    return all(tok in longer for tok in shorter[:2])


async def _fd_fetch_competition(client: httpx.AsyncClient, comp_code: str, date_str: str) -> list:
    """Fetch one competition's matches from football-data.org for a given date."""
    try:
        r = await client.get(
            f"{FOOTBALL_DATA_BASE}/competitions/{comp_code}/matches",
            params={"dateFrom": date_str, "dateTo": date_str},
            headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return r.json().get("matches", [])
    except Exception:
        return []


async def get_football_data_ht_scores(date_str: str) -> list:
    """
    Fetch half-time (and full-time) scores from football-data.org for all
    supported competitions on a given date.

    Returns a list of dicts:
      {home_team, away_team, home_ht, away_ht, home_ft, away_ft, status, competition}
    """
    from src.config import FOOTBALL_DATA_COMPETITIONS
    if not FOOTBALL_DATA_KEY:
        return []
    try:
        async with httpx.AsyncClient() as client:
            raw_lists = await asyncio.gather(
                *[_fd_fetch_competition(client, code, date_str)
                  for code in FOOTBALL_DATA_COMPETITIONS.values()]
            )

        results = []
        seen = set()
        for matches in raw_lists:
            for m in matches:
                key = (m["homeTeam"]["name"], m["awayTeam"]["name"])
                if key in seen:
                    continue
                seen.add(key)
                score   = m.get("score", {})
                ht      = score.get("halfTime", {}) or {}
                ft      = score.get("fullTime", {}) or {}
                home_ht = ht.get("home")
                away_ht = ht.get("away")
                home_ft = ft.get("home")
                away_ft = ft.get("away")
                status  = m.get("status", "")
                if home_ht is None and away_ht is None and home_ft is None and away_ft is None:
                    continue
                results.append({
                    "home_team":   m["homeTeam"]["name"],
                    "away_team":   m["awayTeam"]["name"],
                    "home_ht":     home_ht,
                    "away_ht":     away_ht,
                    "home_ft":     home_ft,
                    "away_ft":     away_ft,
                    "status":      status,
                    "competition": m.get("competition", {}).get("name", ""),
                })
        return results
    except Exception as e:
        print(f"[football-data.org error] {e}")
        return []


def match_ht_to_fixture(fd_matches: list, home_team: str, away_team: str) -> Optional[dict]:
    """Find the football-data.org match entry for an ESPN fixture by team name fuzzy match."""
    for m in fd_matches:
        if _teams_match(m["home_team"], home_team) and _teams_match(m["away_team"], away_team):
            return m
    return None


# ─── TheSportsDB  (final scores, supplemental) ───────────────────────────────

THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"


async def get_thesportsdb_day(date_str: str, sport: str = "Soccer") -> list:
    """
    Fetch all events on a given date from TheSportsDB (free key = 123).
    Returns list of dicts: {home_team, away_team, home_score, away_score, league, event_id}
    Note: HT scores are NOT available on the free tier — only final scores.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{THESPORTSDB_BASE}/{THESPORTSDB_KEY}/eventsday.php",
                params={"d": date_str, "s": sport},
                timeout=15,
            )
            if r.status_code != 200:
                return []
            data = r.json()

        results = []
        for e in (data.get("events") or []):
            home_s = e.get("intHomeScore")
            away_s = e.get("intAwayScore")
            results.append({
                "event_id":   e.get("idEvent", ""),
                "home_team":  e.get("strHomeTeam", ""),
                "away_team":  e.get("strAwayTeam", ""),
                "home_score": int(home_s) if home_s is not None else None,
                "away_score": int(away_s) if away_s is not None else None,
                "league":     e.get("strLeague", ""),
                "date":       e.get("dateEvent", ""),
            })
        return results
    except Exception as e:
        print(f"[TheSportsDB error] {e}")
        return []
