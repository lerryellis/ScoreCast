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
    predict_football_fixture,
)
from src.fetcher import (
    get_espn_team_schedule_raw, get_espn_fixture_dates_for_month,
    get_espn_soccer_fixtures, get_espn_nba_scoreboard,
)
from src.config import ESPN_FOOTBALL_LEAGUES

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
    return {"leagues": list(ESPN_FOOTBALL_LEAGUES.keys())}


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


@app.get("/api/predictions/basketball")
async def basketball_predictions():
    try:
        predictions = await get_all_basketball_predictions()
        return {"sport": "basketball", "matches": predictions}
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
            games = await get_espn_nba_scoreboard()
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "date": date.today().isoformat()}
