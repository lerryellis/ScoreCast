"""
Orchestrator — given a fixture, pulls all data, builds features, runs model.
"""

import asyncio
from datetime import date
from src.fetcher import (
    get_espn_soccer_fixtures, get_espn_team_match_history, get_espn_team_all_matches,
    get_espn_head_to_head, get_espn_team_schedule_raw, get_espn_standings,
    get_intl_team_all_matches, get_intl_head_to_head,
    get_nba_scoreboard, get_nba_team_history,
    get_football_data_ht_scores, match_ht_to_fixture,
)
from src.features.football import build_football_features
from src.features.international import build_international_features
from src.features.basketball import build_basketball_features
from src.models.football_model import predict_football_score
from src.models.basketball_model import predict_basketball_score
from src.config import FOOTBALL_LEAGUES, ESPN_FOOTBALL_LEAGUES, ESPN_INTERNATIONAL_LEAGUES, CUP_TO_LEAGUE


# ─── Football ─────────────────────────────────────────────────────────────────

async def predict_football_fixture(fixture: dict, standings: dict = None,
                                    home_bias: float = 1.0, away_bias: float = 1.0,
                                    rho_factor: float = 1.0, **_) -> dict:
    """Full prediction pipeline for one football fixture."""
    home_id     = fixture["home_team_id"]
    away_id     = fixture["away_team_id"]
    league_slug = fixture.get("league_slug", "eng.1")

    # For cup competitions, use parent league for form data (cup scorelines are unreliable)
    form_slug = CUP_TO_LEAGUE.get(league_slug, league_slug)

    # League matches for form/attack/defence ratings
    # All-competition matches for rest/congestion (includes cups)
    home_matches, away_matches, home_all, away_all, h2h = await asyncio.gather(
        get_espn_team_match_history(home_id, form_slug, n=38),
        get_espn_team_match_history(away_id, form_slug, n=38),
        get_espn_team_all_matches(home_id, form_slug, n=20),
        get_espn_team_all_matches(away_id, form_slug, n=20),
        get_espn_head_to_head(home_id, away_id, league_slug),
    )

    standings = standings or {}
    home_rank  = standings.get(str(home_id), {}).get("rank", 0)
    away_rank  = standings.get(str(away_id), {}).get("rank", 0)

    features = build_football_features(
        home_matches, away_matches, h2h,
        fixture["home_team"], fixture["away_team"],
        home_rank=home_rank, away_rank=away_rank,
        home_all_matches=home_all, away_all_matches=away_all,
    )
    # Apply goal bias calibration from historical prediction errors
    # This single multiplier captures all systematic over/under-prediction
    calibrated_lh = features["lambda_home"] * home_bias
    calibrated_la = features["lambda_away"] * away_bias
    features["lambda_home"] = round(calibrated_lh, 4)
    features["lambda_away"] = round(calibrated_la, 4)
    features["home_bias_applied"]    = round(home_bias, 4)
    features["away_bias_applied"]    = round(away_bias, 4)
    features["rho_factor_applied"]   = round(rho_factor, 4)
    effective_rho = max(-0.15, min(0.0, -0.04 * rho_factor))

    # ── Base Poisson prediction (used as ML input features) ───────────────
    base_pred = predict_football_score(calibrated_lh, calibrated_la, rho=effective_rho)

    # ── ML correction: re-run Poisson with ML-refined lambdas ─────────────
    from src.models.ml_model import get_football_ml
    ml = get_football_ml()
    ml_corrected = False
    if ml and ml.trained:
        ml_row = {
            **features,
            "league":        fixture.get("league", ""),
            "win_prob":      base_pred["win_probability"],
            "draw_prob":     base_pred["draw_probability"],
            "loss_prob":     base_pred["loss_probability"],
            "confidence":    base_pred["confidence"],
            "predicted_home": base_pred["predicted_home"],
            "predicted_away": base_pred["predicted_away"],
            **{k: base_pred["over_under"].get(k) for k in ("over_0_5","over_1_5","over_2_5","over_3_5")},
        }
        result = ml.predict(ml_row)
        if result:
            ml_lh, ml_la = result
            prediction = predict_football_score(ml_lh, ml_la, rho=effective_rho)
            features["ml_lambda_home"] = round(ml_lh, 4)
            features["ml_lambda_away"] = round(ml_la, 4)
            ml_corrected = True
    if not ml_corrected:
        prediction = base_pred

    features["ml_corrected"] = ml_corrected

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
    from datetime import date as _date
    league_slug = ESPN_FOOTBALL_LEAGUES.get(league_name)
    if not league_slug:
        return []

    date_str = target_date or _date.today().isoformat()

    from src.database import get_bias_factors
    fixtures, standings, fd_matches, bias = await asyncio.gather(
        get_espn_soccer_fixtures(league_slug, target_date),
        get_espn_standings(league_slug),
        get_football_data_ht_scores(date_str),
        get_bias_factors(),
    )

    # Pick league-specific calibration if available, else global
    league_bias = bias.get("leagues", {}).get(league_name) or bias.get("global", {})
    home_bias   = league_bias.get("home", 1.0)
    away_bias   = league_bias.get("away", 1.0)
    rho_factor  = league_bias.get("rho_factor", 1.0)

    # Fetch locked predictions for any live/final games — don't re-run the model
    from src.database import get_saved_predictions
    live_final_ids = [f["fixture_id"] for f in fixtures if f.get("is_live") or f.get("is_final")]
    saved_preds = await get_saved_predictions(live_final_ids, date_str) if live_final_ids else {}

    results = []
    for fixture in fixtures:
        try:
            if (fixture.get("is_live") or fixture.get("is_final")) and str(fixture["fixture_id"]) in saved_preds:
                # Game has started or finished — show the locked pre-kickoff prediction
                saved = saved_preds[str(fixture["fixture_id"])]
                pred = {
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
                    "prediction":    {
                        "predicted_home":    saved["predicted_home"],
                        "predicted_away":    saved["predicted_away"],
                        "predicted_home_ht": saved.get("predicted_home_ht"),
                        "predicted_away_ht": saved.get("predicted_away_ht"),
                        "win_probability":   saved.get("win_prob"),
                        "draw_probability":  saved.get("draw_prob"),
                        "loss_probability":  saved.get("loss_prob"),
                        "confidence":        saved.get("confidence"),
                        "safe_bet":          {"line": saved.get("safe_bet_line"), "probability": saved.get("safe_bet_prob")},
                        "over_under":        {k: saved.get(k) for k in ("over_0_5","over_1_5","over_2_5","over_3_5")},
                    },
                    "prediction_locked": True,
                }
                results.append(pred)
                continue

            pred = await predict_football_fixture(
                fixture, standings=standings,
                home_bias=home_bias, away_bias=away_bias,
                rho_factor=rho_factor,
            )

            # Enrich with HT scores from football-data.org (better source than ESPN)
            fd = match_ht_to_fixture(fd_matches, fixture["home_team"], fixture["away_team"])
            if fd:
                if fd.get("home_ht") is not None:
                    pred["home_goals_ht"] = fd["home_ht"]
                    pred["away_goals_ht"] = fd["away_ht"]
                # Also fill in FT score if ESPN didn't get it
                if pred.get("home_goals") is None and fd.get("home_ft") is not None:
                    pred["home_goals"] = fd["home_ft"]
                    pred["away_goals"] = fd["away_ft"]
                if not pred.get("is_final") and fd.get("status") == "FINISHED":
                    pred["is_final"] = True

            results.append(pred)
        except Exception as e:
            print(f"[Football prediction error] {fixture.get('home_team')} vs "
                  f"{fixture.get('away_team')}: {e}")

    # Save predictions to Supabase (fire-and-forget)
    if results:
        try:
            from src.database import save_predictions
            asyncio.create_task(save_predictions(results))
        except Exception:
            pass

    return results


# ─── International Football ──────────────────────────────────────────────────

async def predict_international_fixture(fixture: dict,
                                         home_bias: float = 1.0,
                                         away_bias: float = 1.0,
                                         rho_factor: float = 1.0) -> dict:
    """Full prediction pipeline for one international football fixture."""
    home_id = fixture["home_team_id"]
    away_id = fixture["away_team_id"]

    home_matches, away_matches, h2h = await asyncio.gather(
        get_intl_team_all_matches(home_id, n=20),
        get_intl_team_all_matches(away_id, n=20),
        get_intl_head_to_head(home_id, away_id),
    )

    features = build_international_features(
        home_matches, away_matches, h2h,
        fixture["home_team"], fixture["away_team"],
    )

    calibrated_lh = features["lambda_home"] * home_bias
    calibrated_la = features["lambda_away"] * away_bias
    features["lambda_home"] = round(calibrated_lh, 4)
    features["lambda_away"] = round(calibrated_la, 4)
    features["home_bias_applied"] = round(home_bias, 4)
    features["away_bias_applied"] = round(away_bias, 4)
    features["rho_factor_applied"] = round(rho_factor, 4)
    effective_rho = max(-0.15, min(0.0, -0.04 * rho_factor))
    prediction = predict_football_score(calibrated_lh, calibrated_la, rho=effective_rho)

    return {
        "sport":          "international",
        "fixture_id":     fixture["fixture_id"],
        "home_team":      fixture["home_team"],
        "home_team_id":   fixture["home_team_id"],
        "home_team_logo": fixture.get("home_team_logo", ""),
        "away_team":      fixture["away_team"],
        "away_team_id":   fixture["away_team_id"],
        "away_team_logo": fixture.get("away_team_logo", ""),
        "league":         fixture.get("league", ""),
        "league_slug":    fixture.get("league_slug", ""),
        "match_time":     fixture.get("date", ""),
        "venue":          fixture.get("venue", ""),
        "status":         fixture.get("status", ""),
        "is_live":        fixture.get("is_live", False),
        "is_final":       fixture.get("is_final", False),
        "home_goals":     fixture.get("home_goals"),
        "away_goals":     fixture.get("away_goals"),
        "prediction":     prediction,
        "features":       features,
        "home_form":      [{"date": m["date"], "goals_for": m["goals_for"],
                           "goals_ag": m["goals_ag"], "is_home": m["is_home"]}
                          for m in home_matches[:5]],
        "away_form":      [{"date": m["date"], "goals_for": m["goals_for"],
                           "goals_ag": m["goals_ag"], "is_home": m["is_home"]}
                          for m in away_matches[:5]],
        "h2h":            h2h[:5],
    }


async def get_all_international_predictions(league_name: str = "World Cup 2026",
                                             target_date: str = None) -> list:
    """Fetch fixtures for an international competition and predict all."""
    from datetime import date as _date
    from src.database import get_bias_factors

    league_slug = ESPN_INTERNATIONAL_LEAGUES.get(league_name)
    if not league_slug:
        return []

    fixtures, bias = await asyncio.gather(
        get_espn_soccer_fixtures(league_slug, target_date),
        get_bias_factors(sport="international"),
    )

    intl_bias  = bias.get("global", {})
    home_bias  = intl_bias.get("home", 1.0)
    away_bias  = intl_bias.get("away", 1.0)
    rho_factor = intl_bias.get("rho_factor", 1.0)

    results = []
    for fixture in fixtures:
        try:
            pred = await predict_international_fixture(
                fixture, home_bias=home_bias, away_bias=away_bias,
                rho_factor=rho_factor,
            )
            results.append(pred)
        except Exception as e:
            print(f"[Intl prediction error] {fixture.get('home_team')} vs "
                  f"{fixture.get('away_team')}: {e}")

    if results:
        try:
            from src.database import save_predictions
            asyncio.create_task(save_predictions(results))
        except Exception:
            pass

    return results


# ─── Basketball ───────────────────────────────────────────────────────────────

async def predict_basketball_game(game: dict,
                                   home_bias: float = 1.0,
                                   away_bias: float = 1.0) -> dict:
    """Full prediction pipeline for one NBA game. Uses ESPN + nba_api fallback."""
    home_id   = game.get("home_team_id")
    away_id   = game.get("away_team_id")
    home_name = game.get("home_team", "")
    away_name = game.get("away_team", "")

    async def _empty():
        return []

    home_games, away_games = await asyncio.gather(
        get_nba_team_history(home_id, home_name) if home_id else _empty(),
        get_nba_team_history(away_id, away_name) if away_id else _empty(),
    )

    # H2H: games both teams played (matching game IDs)
    away_ids = {g["game_id"] for g in away_games}
    h2h = []
    for g in home_games:
        if g["game_id"] in away_ids:
            h2h.append({
                "home_team":  game["home_team"],
                "away_team":  game["away_team"],
                "home_goals": g["pts_for"],
                "away_goals": g["pts_ag"],
                "date":       g["date"],
                "playoff":    g.get("playoff", False),
            })

    features   = build_basketball_features(
        home_games, away_games, h2h,
        game["home_team"], game["away_team"],
    )
    # Apply bias calibration from resolved predictions
    features["home_predicted"] = round(features["home_predicted"] * home_bias, 1)
    features["away_predicted"] = round(features["away_predicted"] * away_bias, 1)
    features["home_bias_applied"] = round(home_bias, 4)
    features["away_bias_applied"] = round(away_bias, 4)

    # ── Base basketball prediction (used as ML input features) ────────────
    base_pred = predict_basketball_score(features)

    # ── ML correction: re-run with ML-refined score estimates ─────────────
    from src.models.ml_model import get_basketball_ml
    ml = get_basketball_ml()
    ml_corrected = False
    if ml and ml.trained:
        ml_row = {
            "predicted_home": base_pred["predicted_home"],
            "predicted_away": base_pred["predicted_away"],
            "win_prob":       base_pred["win_probability"],
            "loss_prob":      base_pred["loss_probability"],
            "confidence":     base_pred["confidence"],
            "league":         "NBA",
        }
        result = ml.predict(ml_row)
        if result:
            ml_home, ml_away = result
            features["home_predicted"] = round(ml_home, 1)
            features["away_predicted"] = round(ml_away, 1)
            features["ml_home_predicted"] = round(ml_home, 1)
            features["ml_away_predicted"] = round(ml_away, 1)
            prediction = predict_basketball_score(features)
            ml_corrected = True
    if not ml_corrected:
        prediction = base_pred

    features["ml_corrected"] = ml_corrected

    # Normalise form to same shape frontend expects
    def _fmt_form(games):
        return [{"date": g["date"], "goals_for": g["pts_for"], "goals_ag": g["pts_ag"],
                 "is_home": g.get("is_home", True)} for g in games[:5]]

    return {
        "sport":          "basketball",
        "game_id":        game.get("game_id", ""),
        "home_team":      game["home_team"],
        "home_team_id":   game.get("home_team_id", ""),
        "home_abbr":      game.get("home_abbr", ""),
        "home_team_logo": game.get("home_team_logo", ""),
        "away_team":      game["away_team"],
        "away_team_id":   game.get("away_team_id", ""),
        "away_abbr":      game.get("away_abbr", ""),
        "away_team_logo": game.get("away_team_logo", ""),
        "status":         game.get("status", ""),
        "is_live":        game.get("is_live", False),
        "is_final":       game.get("is_final", False),
        "home_score":     game.get("home_score"),
        "away_score":     game.get("away_score"),
        "home_form":      _fmt_form(home_games),
        "away_form":      _fmt_form(away_games),
        "h2h":            h2h[:5],
        "prediction":     prediction,
        "features":       features,
    }


async def get_all_basketball_predictions(target_date: str = None) -> list:
    """Fetch NBA games (ESPN + nba_api fallback) for a given date and predict all."""
    from datetime import date as _date
    from src.database import get_bias_factors

    games, bias = await asyncio.gather(
        get_nba_scoreboard(target_date),
        get_bias_factors(sport="basketball"),
    )
    if not games:
        return []

    nba_bias   = bias.get("leagues", {}).get("NBA") or bias.get("global", {})
    home_bias  = nba_bias.get("home", 1.0)
    away_bias  = nba_bias.get("away", 1.0)
    match_date = target_date or _date.today().isoformat()

    # Fetch locked predictions for live/final games
    from src.database import get_saved_predictions
    live_final_game_ids = [g["game_id"] for g in games if g.get("is_live") or g.get("is_final")]
    saved_preds = await get_saved_predictions(live_final_game_ids, match_date) if live_final_game_ids else {}

    async def _safe_predict(game):
        # Game has started — return locked pre-kickoff prediction
        if (game.get("is_live") or game.get("is_final")) and str(game["game_id"]) in saved_preds:
            saved = saved_preds[str(game["game_id"])]
            return {
                "sport":          "basketball",
                "game_id":        game["game_id"],
                "home_team":      game["home_team"],
                "home_team_id":   game.get("home_team_id", ""),
                "home_abbr":      game.get("home_abbr", ""),
                "home_team_logo": game.get("home_team_logo", ""),
                "away_team":      game["away_team"],
                "away_team_id":   game.get("away_team_id", ""),
                "away_abbr":      game.get("away_abbr", ""),
                "away_team_logo": game.get("away_team_logo", ""),
                "status":         game.get("status", ""),
                "is_live":        game.get("is_live", False),
                "is_final":       game.get("is_final", False),
                "home_score":     game.get("home_score"),
                "away_score":     game.get("away_score"),
                "prediction": {
                    "predicted_home":  saved["predicted_home"],
                    "predicted_away":  saved["predicted_away"],
                    "win_probability": saved.get("win_prob"),
                    "loss_probability":saved.get("loss_prob"),
                    "confidence":      saved.get("confidence"),
                    "safe_bet": {"line": saved.get("safe_bet_line"), "probability": saved.get("safe_bet_prob")},
                },
                "match_date":       match_date,
                "prediction_locked": True,
            }
        try:
            return await predict_basketball_game(game, home_bias=home_bias, away_bias=away_bias)
        except Exception as e:
            print(f"[Basketball prediction error] {game.get('home_team')} vs "
                  f"{game.get('away_team')}: {e}")
            return None

    preds = await asyncio.gather(*[_safe_predict(g) for g in games])
    results = [p for p in preds if p is not None]

    for p in results:
        p["match_date"] = match_date[:10]

    if results:
        try:
            from src.database import save_basketball_predictions
            asyncio.create_task(save_basketball_predictions(results))
        except Exception:
            pass

    return results
