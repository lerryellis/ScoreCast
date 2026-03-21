"""
BetScore FastAPI backend.
Serves predictions to the frontend and runs background refresh.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
import asyncio

from src.predictor import (
    get_all_football_predictions,
    get_all_basketball_predictions,
)
from src.config import FOOTBALL_LEAGUES

app = FastAPI(title="BetScore", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── In-memory cache ──────────────────────────────────────────────────────────
_cache = {
    "football":   {"data": [], "fetched_at": None},
    "basketball": {"data": [], "fetched_at": None},
}


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/api/leagues")
async def list_leagues():
    return {"leagues": list(FOOTBALL_LEAGUES.keys())}


@app.get("/api/predictions/football")
async def football_predictions(
    league: str = Query("Premier League"),
    date:   str = Query(None),
):
    try:
        predictions = await get_all_football_predictions(
            league_name=league,
            season=2024,
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "date": date.today().isoformat()}
