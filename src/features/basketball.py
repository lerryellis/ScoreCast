"""
Basketball-specific feature engineering.
Builds the feature vector fed into the XGBoost regressor.
"""

import numpy as np
from src.features.base import (
    rolling_average, days_since_last_match,
    rest_factor, h2h_avg_scores, injury_impact_factor
)
from src.config import BASKETBALL_FORM_WINDOW

NBA_AVG_POINTS = 113.0   # approximate NBA league average per game


def build_basketball_features(
    home_games:     list,
    away_games:     list,
    h2h:            list,
    home_team_name: str,
    away_team_name: str,
    home_injuries:  list = None,
    away_injuries:  list = None,
) -> dict:
    """
    Returns a feature dict consumed by the basketball prediction model.
    Each team gets a predicted points total from an XGBoost regressor.
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    def _extract(games, key):
        return [g[key] for g in games if g.get(key) is not None]

    def _weighted_avg(games, key, window):
        """Weighted rolling average — playoff games count 1.5x regular season."""
        vals = [(g[key], 1.5 if g.get("playoff") else 1.0)
                for g in games[:window] if g.get(key) is not None]
        if not vals:
            return None
        total_w = sum(w for _, w in vals)
        return sum(v * w for v, w in vals) / total_w

    # ── Scoring averages (playoff games weighted 1.5×) ────────────────────
    home_avg_pts   = _weighted_avg(home_games, "pts_for", BASKETBALL_FORM_WINDOW) or NBA_AVG_POINTS
    home_avg_allow = _weighted_avg(home_games, "pts_ag",  BASKETBALL_FORM_WINDOW) or NBA_AVG_POINTS
    away_avg_pts   = _weighted_avg(away_games, "pts_for", BASKETBALL_FORM_WINDOW) or NBA_AVG_POINTS
    away_avg_allow = _weighted_avg(away_games, "pts_ag",  BASKETBALL_FORM_WINDOW) or NBA_AVG_POINTS

    # ── Offensive / Defensive ratings (simplified) ────────────────────────
    home_off_rating = home_avg_pts   / NBA_AVG_POINTS
    home_def_rating = home_avg_allow / NBA_AVG_POINTS   # lower = better defence
    away_off_rating = away_avg_pts   / NBA_AVG_POINTS
    away_def_rating = away_avg_allow / NBA_AVG_POINTS

    # ── Shooting efficiency ───────────────────────────────────────────────
    home_fg    = rolling_average(_extract(home_games, "fg_pct"),  BASKETBALL_FORM_WINDOW)
    home_fg3   = rolling_average(_extract(home_games, "fg3_pct"), BASKETBALL_FORM_WINDOW)
    away_fg    = rolling_average(_extract(away_games, "fg_pct"),  BASKETBALL_FORM_WINDOW)
    away_fg3   = rolling_average(_extract(away_games, "fg3_pct"), BASKETBALL_FORM_WINDOW)

    # ── Turnovers (negative feature) ──────────────────────────────────────
    home_tov   = rolling_average(_extract(home_games, "tov"), BASKETBALL_FORM_WINDOW)
    away_tov   = rolling_average(_extract(away_games, "tov"), BASKETBALL_FORM_WINDOW)

    # ── Rest / back-to-back ────────────────────────────────────────────────
    home_dates  = [g["date"] for g in home_games]
    away_dates  = [g["date"] for g in away_games]
    home_rest   = rest_factor(days_since_last_match(home_dates))
    away_rest   = rest_factor(days_since_last_match(away_dates))

    # ── Injury impact ──────────────────────────────────────────────────────
    home_inj    = injury_impact_factor(home_injuries, sport="basketball")
    away_inj    = injury_impact_factor(away_injuries, sport="basketball")

    # ── H2H ────────────────────────────────────────────────────────────────
    home_h2h    = h2h_avg_scores(h2h, home_team_name)
    away_h2h    = h2h_avg_scores(h2h, away_team_name)

    # ── Predicted base scores ───────────────────────────────────────────────
    # Simple interaction: offensive rating vs opponent defensive rating
    # XGBoost will refine this but we build a useful prior
    home_predicted = (home_off_rating * away_def_rating * NBA_AVG_POINTS
                      * home_rest * home_inj * 1.03)   # home court ~3 pts
    away_predicted = (away_off_rating * home_def_rating * NBA_AVG_POINTS
                      * away_rest * away_inj)

    if home_h2h["matches"] >= 2:
        # Increase H2H weight during playoffs — series games are highly predictive
        playoff_h2h = sum(1 for g in h2h if g.get("playoff"))
        h2h_weight  = 0.30 if playoff_h2h >= 1 else 0.15
        home_predicted = (1 - h2h_weight) * home_predicted + h2h_weight * home_h2h["avg_for"]
        away_predicted = (1 - h2h_weight) * away_predicted + h2h_weight * away_h2h["avg_for"]

    return {
        # Target predictions (refined by model)
        "home_predicted":    round(home_predicted, 1),
        "away_predicted":    round(away_predicted, 1),
        # Features
        "home_off_rating":   round(home_off_rating,  3),
        "home_def_rating":   round(home_def_rating,  3),
        "away_off_rating":   round(away_off_rating,  3),
        "away_def_rating":   round(away_def_rating,  3),
        "home_fg_pct":       round(home_fg,   3),
        "home_fg3_pct":      round(home_fg3,  3),
        "away_fg_pct":       round(away_fg,   3),
        "away_fg3_pct":      round(away_fg3,  3),
        "home_tov_avg":      round(home_tov,  1),
        "away_tov_avg":      round(away_tov,  1),
        "home_rest_factor":  round(home_rest, 3),
        "away_rest_factor":  round(away_rest, 3),
        "home_injury_factor":round(home_inj,  3),
        "away_injury_factor":round(away_inj,  3),
        "h2h_matches":       home_h2h["matches"],
        "h2h_home_avg":      round(home_h2h["avg_for"], 1),
        "h2h_away_avg":      round(away_h2h["avg_for"], 1),
    }
