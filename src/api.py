"""
ScoreCast FastAPI backend.
Serves predictions to the frontend and runs background refresh.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
import asyncio

from src.predictor import (
    get_all_football_predictions,
    get_all_basketball_predictions,
    get_all_international_predictions,
    predict_football_fixture,
)
from src.fetcher import (
    get_espn_team_schedule_raw, get_espn_fixture_dates_for_month,
    get_espn_soccer_fixtures, get_espn_nba_scoreboard, get_nba_scoreboard,
    get_football_data_ht_scores, get_thesportsdb_day,
    get_espn_nba_dates_for_month, get_espn_nba_full_team_schedule,
)
from src.config import ESPN_FOOTBALL_LEAGUES, ESPN_INTERNATIONAL_LEAGUES

app = FastAPI(title="ScoreCast", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/api/leagues")
async def list_leagues():
    return {
        "leagues": list(ESPN_FOOTBALL_LEAGUES.keys()),
        "international": list(ESPN_INTERNATIONAL_LEAGUES.keys()),
    }


@app.get("/api/predictions/football")
async def football_predictions(
    league: str = Query("Premier League"),
    date:   str = Query(None),
):
    try:
        predictions = await get_all_football_predictions(
            league_name=league,
            target_date=date,
        )
        return {"sport": "football", "league": league, "matches": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/international-leagues")
async def list_international_leagues():
    return {"leagues": list(ESPN_INTERNATIONAL_LEAGUES.keys())}


@app.get("/api/predictions/international")
async def international_predictions(
    league: str = Query("World Cup 2026"),
    date:   str = Query(None),
):
    try:
        predictions = await get_all_international_predictions(
            league_name=league, target_date=date,
        )
        return {"sport": "international", "league": league, "matches": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/international/fixture-dates")
async def intl_fixture_dates(
    league: str = Query("World Cup 2026"),
    year:  int = Query(None),
    month: int = Query(None),
):
    from datetime import date as _date
    slug = ESPN_INTERNATIONAL_LEAGUES.get(league)
    if not slug:
        return {"dates": []}
    today = _date.today()
    y = year or today.year
    m = month or today.month
    try:
        dates = await get_espn_fixture_dates_for_month(slug, y, m)
        return {"year": y, "month": m, "dates": dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/basketball")
async def basketball_predictions(date: str = Query(None)):
    try:
        predictions = await get_all_basketball_predictions(target_date=date)
        return {"sport": "basketball", "matches": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/basketball/game-dates")
async def nba_game_dates(year: int = Query(None), month: int = Query(None)):
    from datetime import date as _date
    today = _date.today()
    y = year or today.year
    m = month or today.month
    try:
        dates = await get_espn_nba_dates_for_month(y, m)
        return {"year": y, "month": m, "dates": dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/basketball/team-schedule")
async def nba_team_schedule(team_id: str = Query(...)):
    try:
        from src.predictor import predict_basketball_game
        events = await get_espn_nba_full_team_schedule(team_id)
        results = []
        for event in events:
            comp = event["competitions"][0]
            status = comp.get("status", {}).get("type", {})
            completed = status.get("completed", False)
            competitors = comp["competitors"]
            home = next((c for c in competitors if c["homeAway"] == "home"), None)
            away = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home or not away:
                continue
            entry = {
                "game_id":      event["id"],
                "date":         event["date"],
                "home_team":    home["team"]["displayName"],
                "home_team_id": home["team"]["id"],
                "home_abbr":    home["team"].get("abbreviation", ""),
                "away_team":    away["team"]["displayName"],
                "away_team_id": away["team"]["id"],
                "away_abbr":    away["team"].get("abbreviation", ""),
                "completed":    completed,
                "status_text":  status.get("shortDetail", ""),
            }
            if completed:
                hs = home.get("score", {})
                as_ = away.get("score", {})
                entry["home_score"] = int(hs.get("value", 0) if isinstance(hs, dict) else (hs or 0))
                entry["away_score"] = int(as_.get("value", 0) if isinstance(as_, dict) else (as_ or 0))
            else:
                try:
                    pred = await predict_basketball_game(entry)
                    entry["prediction"] = pred["prediction"]
                except Exception:
                    pass
            results.append(entry)
        return {"team_id": team_id, "matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/football/team-schedule")
async def team_schedule(
    team_id:     str = Query(...),
    league_slug: str = Query(...),
):
    """Return a team's full season schedule with predictions for upcoming games."""
    try:
        events = await get_espn_team_schedule_raw(team_id, league_slug)
        league_name = next(
            (k for k, v in ESPN_FOOTBALL_LEAGUES.items() if v == league_slug), league_slug
        )
        results = []
        for event in events:
            comp = event["competitions"][0]
            status = comp.get("status", {}).get("type", {})
            completed = status.get("completed", False)
            competitors = comp["competitors"]
            home = next((c for c in competitors if c["homeAway"] == "home"), None)
            away = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home or not away:
                continue

            entry = {
                "fixture_id":   event["id"],
                "date":         event["date"],
                "home_team":    home["team"]["displayName"],
                "home_team_id": home["team"]["id"],
                "away_team":    away["team"]["displayName"],
                "away_team_id": away["team"]["id"],
                "venue":        comp.get("venue", {}).get("fullName", ""),
                "league":       league_name,
                "league_slug":  league_slug,
                "completed":    completed,
                "status_text":  status.get("shortDetail", ""),
            }

            if completed:
                hs = home.get("score", {})
                as_ = away.get("score", {})
                entry["home_goals"] = int(hs.get("value", 0)) if isinstance(hs, dict) else (hs or 0)
                entry["away_goals"] = int(as_.get("value", 0)) if isinstance(as_, dict) else (as_ or 0)
            else:
                try:
                    pred = await predict_football_fixture(entry)
                    entry["prediction"] = pred["prediction"]
                except Exception:
                    pass

            results.append(entry)

        return {"team_id": team_id, "league_slug": league_slug, "matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/football/fixture-dates")
async def fixture_dates(
    league: str = Query("Premier League"),
    year:   int = Query(None),
    month:  int = Query(None),
):
    """Return list of dates that have fixtures for a league in a given month."""
    from datetime import date as _date
    today = _date.today()
    y = year  or today.year
    m = month or today.month
    league_slug = ESPN_FOOTBALL_LEAGUES.get(league)
    if not league_slug:
        return {"dates": []}
    try:
        dates = await get_espn_fixture_dates_for_month(league_slug, y, m)
        return {"league": league, "year": y, "month": m, "dates": dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live-scores")
async def live_scores():
    """Return today's in-progress and recently completed scores across all leagues + NBA."""
    today = date.today().isoformat()

    async def _football_scores(league_name: str, league_slug: str):
        try:
            fixtures = await get_espn_soccer_fixtures(league_slug, today)
            items = []
            for f in fixtures:
                is_live  = f.get("is_live", False)
                is_final = f.get("is_final", False)
                if not is_live and not is_final:
                    continue
                items.append({
                    "sport":      "football",
                    "league":     league_name,
                    "home_team":  f.get("home_team", ""),
                    "away_team":  f.get("away_team", ""),
                    "home_score": f.get("home_goals"),
                    "away_score": f.get("away_goals"),
                    "status":     f.get("status", ""),
                    "is_live":    is_live,
                    "is_final":   is_final,
                })
            return items
        except Exception:
            return []

    async def _nba_scores():
        try:
            games = await get_nba_scoreboard()
            items = []
            for g in games:
                is_live  = g.get("is_live", False)
                is_final = g.get("is_final", False)
                if not is_live and not is_final:
                    continue
                items.append({
                    "sport":      "basketball",
                    "league":     "NBA",
                    "home_team":  g.get("home_abbr") or g.get("home_team", ""),
                    "away_team":  g.get("away_abbr") or g.get("away_team", ""),
                    "home_score": g.get("home_score"),
                    "away_score": g.get("away_score"),
                    "status":     g.get("status", ""),
                    "is_live":    is_live,
                    "is_final":   is_final,
                })
            return items
        except Exception:
            return []

    tasks = [_football_scores(name, slug) for name, slug in ESPN_FOOTBALL_LEAGUES.items()]
    tasks.append(_nba_scores())
    results = await asyncio.gather(*tasks)
    scores = [item for group in results for item in group]
    return {"scores": scores}


@app.get("/api/football/ht-scores")
async def football_ht_scores(date: str = Query(None)):
    """Return HT scores from football-data.org for all matches on a given date."""
    from datetime import date as _date
    d = date or _date.today().isoformat()
    try:
        matches = await get_football_data_ht_scores(d)
        return {"date": d, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scores/day")
async def scores_by_day(
    date:  str = Query(None),
    sport: str = Query("Soccer"),
):
    """Return TheSportsDB event scores for a given date and sport."""
    from datetime import date as _date
    d = date or _date.today().isoformat()
    try:
        events = await get_thesportsdb_day(d, sport)
        return {"date": d, "sport": sport, "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/day")
async def predictions_day(date: str = Query(None), sport: str = Query("football")):
    """Return all predictions for a given date, with actual scores where resolved."""
    from datetime import date as _date
    from src.config import SUPABASE_URL, SUPABASE_KEY
    d = date or _date.today().isoformat()
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"date": d, "matches": []}
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)

        def _fetch():
            rows = (
                client.table("predictions")
                      .select("*, prediction_results(*)")
                      .eq("match_date", d)
                      .eq("sport", sport)
                      .order("home_team")
                      .execute()
            )
            return rows.data or []

        rows = await asyncio.to_thread(_fetch)
        matches = []
        for r in rows:
            result = (r.get("prediction_results") or [None])[0]
            matches.append({
                "fixture_id":      r["fixture_id"],
                "league":          r.get("league", ""),
                "home_team":       r["home_team"],
                "away_team":       r["away_team"],
                "predicted_home":  r["predicted_home"],
                "predicted_away":  r["predicted_away"],
                "win_prob":        r.get("win_prob"),
                "draw_prob":       r.get("draw_prob"),
                "loss_prob":       r.get("loss_prob"),
                "confidence":      r.get("confidence"),
                "safe_bet_line":   r.get("safe_bet_line"),
                "safe_bet_prob":   r.get("safe_bet_prob"),
                "actual_home":     result["actual_home"]        if result else None,
                "actual_away":     result["actual_away"]        if result else None,
                "outcome_correct": result["outcome_correct"]    if result else None,
                "exact_correct":   result["exact_correct"]      if result else None,
                "home_error":      result["home_error"]         if result else None,
                "away_error":      result["away_error"]         if result else None,
                "safe_bet_correct": result["safe_bet_correct"]  if result else None,
            })
        return {"date": d, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/accuracy/trend")
async def accuracy_trend(sport: str = Query("football")):
    """Daily + 7-day rolling accuracy stats for the performance graph."""
    try:
        from src.database import get_accuracy_trend
        return await get_accuracy_trend(sport=sport)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scorecard")
async def scorecard(sport: str = Query("football")):
    """Public accuracy scorecard — aggregated prediction vs actual stats."""
    try:
        from src.database import get_scorecard
        data = await get_scorecard(sport=sport)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/resolve")
async def resolve_predictions_endpoint(admin_key: str = Query(...)):
    """Resolve yesterday's unresolved predictions against actual scores."""
    from src.config import ADMIN_KEY
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        from src.database import resolve_predictions
        count = await resolve_predictions()
        return {"resolved": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "date": date.today().isoformat()}
