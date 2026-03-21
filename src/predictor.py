"""
Orchestrator — given a fixture, pulls all data, builds features, runs model.
"""

import asyncio
from datetime import date
from src.fetcher import (
    get_football_fixtures, get_team_last_n_matches,
    get_head_to_head_football, get_team_injuries_football,
    get_nba_today_scoreboard, get_nba_team_last_n_games, get_nba_all_teams,
)
from src.features.football import build_football_features
from src.features.basketball import build_basketball_features
from src.models.football_model import predict_football_score
from src.models.basketball_model import predict_basketball_score
from src.config import FOOTBALL_LEAGUES


# ─── Football ─────────────────────────────────────────────────────────────────

async def predict_football_fixture(fixture: dict, season: int = 2024) -> dict:
    """Full prediction pipeline for one football fixture."""
    home_id   = fixture["home_team_id"]
    away_id   = fixture["away_team_id"]
    fix_id    = fixture["fixture_id"]

    # Fetch in parallel
    home_matches, away_matches, h2h, home_inj, away_inj = await asyncio.gather(
        get_team_last_n_matches(home_id, season),
        get_team_last_n_matches(away_id, season),
        get_head_to_head_football(home_id, away_id),
        get_team_injuries_football(home_id, fix_id),
        get_team_injuries_football(away_id, fix_id),
    )

    features = build_football_features(
        home_matches, away_matches, h2h,
        fixture["home_team"], fixture["away_team"],
        home_inj, away_inj,
    )
    prediction = predict_football_score(features["lambda_home"], features["lambda_away"])

    return {
        "sport":       "football",
        "fixture_id":  fix_id,
        "home_team":   fixture["home_team"],
        "away_team":   fixture["away_team"],
        "league":      fixture.get("league", ""),
        "match_time":  fixture.get("date", ""),
        "venue":       fixture.get("venue", ""),
        "prediction":  prediction,
        "features":    features,
    }


async def get_all_football_predictions(league_name: str = "Premier League",
                                        season: int = 2024,
                                        target_date: str = None) -> list:
    """Fetch today's fixtures for a league and predict all of them."""
    if league_name not in FOOTBALL_LEAGUES:
        return []

    league_id = FOOTBALL_LEAGUES[league_name]["id"]
    fixtures  = await get_football_fixtures(league_id, season, target_date)

    results = []
    for fixture in fixtures:
        try:
            pred = await predict_football_fixture(fixture, season)
            results.append(pred)
        except Exception as e:
            print(f"[Football prediction error] {fixture.get('home_team')} vs "
                  f"{fixture.get('away_team')}: {e}")
    return results


# ─── Basketball ───────────────────────────────────────────────────────────────

def predict_basketball_game(game: dict) -> dict:
    """Full prediction pipeline for one NBA game."""
    all_teams  = get_nba_all_teams()
    team_map   = {t["nickname"]: t["id"] for t in all_teams}

    home_id    = team_map.get(game["home_team"])
    away_id    = team_map.get(game["away_team"])

    home_games = get_nba_team_last_n_games(home_id) if home_id else []
    away_games = get_nba_team_last_n_games(away_id) if away_id else []

    # Build h2h from overlapping game ids (simplified)
    h2h = []

    features   = build_basketball_features(
        home_games, away_games, h2h,
        game["home_team"], game["away_team"],
    )
    prediction = predict_basketball_score(features)

    return {
        "sport":       "basketball",
        "game_id":     game.get("game_id", ""),
        "home_team":   game["home_team"],
        "home_abbr":   game.get("home_abbr", ""),
        "away_team":   game["away_team"],
        "away_abbr":   game.get("away_abbr", ""),
        "status":      game.get("status", ""),
        "prediction":  prediction,
        "features":    features,
    }


def get_all_basketball_predictions() -> list:
    """Fetch today's NBA games and predict all of them."""
    games   = get_nba_today_scoreboard()
    results = []
    for game in games:
        try:
            pred = predict_basketball_game(game)
            results.append(pred)
        except Exception as e:
            print(f"[Basketball prediction error] {game.get('home_team')} vs "
                  f"{game.get('away_team')}: {e}")
    return results
