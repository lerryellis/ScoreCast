# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env          # then add API keys
uvicorn src.api:app --reload --port 8000   # dev server → localhost:8000
```

## Architecture

BetScore is a sports score prediction app covering football/soccer and NBA basketball.

**Backend (FastAPI — `src/`)**
- `api.py` — FastAPI app, serves `index.html` + exposes `/api/predictions/football` and `/api/predictions/basketball`
- `predictor.py` — orchestrates: fixture → features → model → prediction dict
- `fetcher.py` — async HTTP calls to API-Football (football) and `nba_api` package (basketball)
- `config.py` — API keys, league IDs, model constants

**Feature Engineering (`src/features/`)**
- `base.py` — shared helpers: rolling averages, rest factor, H2H, injury impact
- `football.py` — attack/defence strength ratings → λ_home, λ_away (expected goals)
- `basketball.py` — offensive/defensive ratings, pace, fatigue → predicted points

**Models (`src/models/`)**
- `football_model.py` — Dixon-Coles Poisson: builds 9×9 scoreline probability matrix, returns most likely scoreline + top 5 + win/draw/loss %
- `basketball_model.py` — score prediction with confidence intervals from scoring variance

**Frontend (`index.html`)**
- Pure HTML/CSS/JS — no build step
- Dark sports-themed UI, two tabs (Football / Basketball)
- Calls `/api/predictions/*` endpoints, renders match cards with scores, probability bars, scorelines

## Key Environment Variables
```
API_FOOTBALL_KEY    # from api-football.com (free: 100 req/day)
ADMIN_KEY           # optional
PORT                # default 8000
```

## Data Sources
- Football: api-football.com (v3) — fixtures, results, H2H, injuries
- Basketball: `nba_api` Python package — free, scrapes NBA.com, no key needed

## Deployment (later)
- Railway: `Procfile` already configured (`uvicorn src.api:app --host 0.0.0.0 --port $PORT`)
- Vercel: not applicable (Python backend) — serve `index.html` from Railway directly
