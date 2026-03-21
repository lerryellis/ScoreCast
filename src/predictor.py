"""
Orchestrator — given a fixture, pulls all data, builds features, runs model.
"""

import asyncio
from datetime import date
from src.fetcher import (
    get_espn_soccer_fixtures, get_espn_team_match_history, get_espn_head_to_head,
    get_espn_team_schedule_raw, get_espn_standings,
    get_espn_nba_scoreboard, get_espn_nba_team_games,
)
from src.features.football import build_football_features
from src.features.basketball import build_basketball_features
from src.models.football_model import predict_football_score
from src.models.basketball_model import predict_basketball_score
from src.config import FOOTBALL_LEAGUES, ESPN_FOOTBALL_LEAGUES


# ─── Football ─────────────────────────────────────────────────────────────────

async def predict_football_fixture(fixture: dict, standings: dict = None, **_) -> dict:
    """Full prediction pipeline for one football fixture."""
    home_id     = fixture["home_team_id"]
    away_id     = fixture["away_team_id"]
    league_slug = fixture.get("league_slug", "eng.1")

    home_matches, away_matches, h2h = await asyncio.gather(
        get_espn_team_match_history(home_id, league_slug, n=38),
        get_espn_team_match_history(away_id, league_slug, n=38),
        get_espn_head_to_head(home_id, away_id, league_slug),
    )

    standings = standings or {}
    home_rank  = standings.get(str(home_id), {}).get("rank", 0)
    away_rank  = standings.get(str(away_id), {}).get("rank", 0)

    features = build_football_features(
        home_matches, away_matches, h2h,
        fixture["home_team"], fixture["away_team"],
        home_rank=home_rank, away_rank=away_rank,
    )
    prediction = predict_football_score(features["lambda_home"], features["lambda_away"])

    return {
        "sport":         "football",
        "fixture_id":    fixture["fixture_id"],
        "home_team":     fixture["home_team"],
        "home_team_id":  fixture["home_team_id"],
        "home_team_logo": fixture.get("home_team_logo", ""),
        "away_team":     fixture["away_team"],
        "away_team_id":  fixture["away_team_id"],
        "away_team_logo": fixture.get("away_team_logo", ""),
        "league":        fixture.get("league", ""),
        "league_slug":   fixture.get("league_slug", ""),
        "match_time":    fixture.get("date", ""),
        "venue":         fixture.get("venue", ""),
        "status":        fixture.get("status", ""),
        "is_live":       fixture.get("is_live", False),
        "is_final":      fixture.get("is_final", False),
        "home_goals":    fixture.get("home_goals"),
        "away_goals":    fixture.get("away_goals"),
        "home_goals_ht": fixture.get("home_goals_ht"),
        "away_goals_ht": fixture.get("away_goals_ht"),
        "prediction":    prediction,
        "features":      features,
        "home_form":     home_matches[:5],
        "away_form":     away_matches[:5],
        "h2h":           h2h[:5],
    }


async def get_all_football_predictions(league_name: str = "Premier League",
                                        season: int = None,
                                        target_date: str = None) -> list:
    """Fetch today's fixtures for a league via ESPN and predict all of them."""
    league_slug = ESPN_FOOTBALL_LEAGUES.get(league_name)
    if not league_slug:
        return []

    fixtures, standings = await asyncio.gather(
        get_espn_soccer_fixtures(league_slug, target_date),
        get_espn_standings(league_slug),
    )

    results = []
    for fixture in fixtures:
        try:
            pred = await predict_football_fixture(fixture, standings=standings)
            results.append(pred)
        except Exception as e:
            print(f"[Football prediction error] {fixture.get('home_team')} vs "
                  f"{fixture.get('away_team')}: {e}")
    return results


# ─── Basketball ───────────────────────────────────────────────────────────────

async def predict_basketball_game(game: dict) -> dict:
    """Full prediction pipeline for one NBA game using ESPN data."""
    home_id = game.get("home_team_id")
    away_id = game.get("away_team_id")

    async def _empty():
        return []

    home_games, away_games = await asyncio.gather(
        get_espn_nba_team_games(home_id) if home_id else _empty(),
        get_espn_nba_team_games(away_id) if away_id else _empty(),
    )

    # H2H: games both teams played (matching game IDs)
    away_ids = {g["game_id"] for g in away_games}
    h2h = []
    for g in home_games:
        if g["game_id"] in away_ids:
            h2h.append({
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_goals": g["pts_for"],
                "away_goals": g["pts_ag"],
                "date": g["date"],
            })

    features   = build_basketball_features(
        home_games, away_games, h2h,
        game["home_team"], game["away_team"],
    )
    prediction = predict_basketball_score(features)

    return {
        "sport":          "basketball",
        "game_id":        game.get("game_id", ""),
        "home_team":      game["home_team"],
        "home_abbr":      game.get("home_abbr", ""),
        "home_team_logo": game.get("home_team_logo", ""),
        "away_team":      game["away_team"],
        "away_abbr":      game.get("away_abbr", ""),
        "away_team_logo": game.get("away_team_logo", ""),
        "status":         game.get("status", ""),
        "prediction":     prediction,
        "features":       features,
    }


async def get_all_basketball_predictions() -> list:
    """Fetch today's NBA games from ESPN and predict all of them."""
    games = await get_espn_nba_scoreboard()
    results = []
    for game in games:
        try:
            pred = await predict_basketball_game(game)
            results.append(pred)
        except Exception as e:
            print(f"[Basketball prediction error] {game.get('home_team')} vs "
                  f"{game.get('away_team')}: {e}")
    return results
