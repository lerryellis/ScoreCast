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
from src.fetcher import get_espn_team_schedule_raw
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
        predictions = get_all_basketball_predictions()
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "date": date.today().isoformat()}
