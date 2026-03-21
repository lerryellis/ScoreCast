"""
Football-specific feature engineering.
Builds the attack/defence strength ratings used by the Dixon-Coles model.
"""

import numpy as np
from src.features.base import (
    rolling_average, days_since_last_match,
    rest_factor, h2h_avg_scores, injury_impact_factor
)
from src.config import FOOTBALL_FORM_WINDOW, HOME_ADVANTAGE_FACTOR


# League average goals per game (used as baseline — updated per league)
LEAGUE_AVG_GOALS = 1.35   # roughly correct for top European leagues


def build_football_features(
    home_matches:   list,
    away_matches:   list,
    h2h:            list,
    home_team_name: str,
    away_team_name: str,
    home_injuries:  list = None,
    away_injuries:  list = None,
) -> dict:
    """
    Returns a feature dict consumed by the football prediction model.

    attack_strength  = team's avg goals scored / league average
    defence_weakness = team's avg goals conceded / league average
    (values > 1.0 mean above average)
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    # ── Goals scored / conceded rolling averages ──────────────────────────
    home_scored   = [m["goals_for"] for m in home_matches if m["goals_for"] is not None]
    home_conceded = [m["goals_ag"]  for m in home_matches if m["goals_ag"]  is not None]
    away_scored   = [m["goals_for"] for m in away_matches if m["goals_for"] is not None]
    away_conceded = [m["goals_ag"]  for m in away_matches if m["goals_ag"]  is not None]

    home_avg_scored   = rolling_average(home_scored,   FOOTBALL_FORM_WINDOW) or LEAGUE_AVG_GOALS
    home_avg_conceded = rolling_average(home_conceded, FOOTBALL_FORM_WINDOW) or LEAGUE_AVG_GOALS
    away_avg_scored   = rolling_average(away_scored,   FOOTBALL_FORM_WINDOW) or LEAGUE_AVG_GOALS
    away_avg_conceded = rolling_average(away_conceded, FOOTBALL_FORM_WINDOW) or LEAGUE_AVG_GOALS

    # ── Strength ratings ──────────────────────────────────────────────────
    home_attack   = home_avg_scored   / LEAGUE_AVG_GOALS
    home_defence  = home_avg_conceded / LEAGUE_AVG_GOALS   # higher = leakier
    away_attack   = away_avg_scored   / LEAGUE_AVG_GOALS
    away_defence  = away_avg_conceded / LEAGUE_AVG_GOALS

    # ── Rest / fatigue ────────────────────────────────────────────────────
    home_dates   = [m["date"] for m in home_matches]
    away_dates   = [m["date"] for m in away_matches]
    home_rest    = rest_factor(days_since_last_match(home_dates))
    away_rest    = rest_factor(days_since_last_match(away_dates))

    # ── Injury impact ─────────────────────────────────────────────────────
    home_inj     = injury_impact_factor(home_injuries)
    away_inj     = injury_impact_factor(away_injuries)

    # ── H2H ───────────────────────────────────────────────────────────────
    home_h2h     = h2h_avg_scores(h2h, home_team_name)
    away_h2h     = h2h_avg_scores(h2h, away_team_name)

    # ── Expected goals (λ) for each team ─────────────────────────────────
    # Dixon-Coles formula:
    #   λ_home = home_attack × away_defence × league_avg × home_advantage
    #   λ_away = away_attack × home_defence × league_avg
    lambda_home = (
        home_attack * away_defence * LEAGUE_AVG_GOALS
        * HOME_ADVANTAGE_FACTOR * home_rest * home_inj
    )
    lambda_away = (
        away_attack * home_defence * LEAGUE_AVG_GOALS
        * away_rest * away_inj
    )

    # Blend in H2H if we have enough matches
    if home_h2h["matches"] >= 3:
        h2h_weight  = 0.20
        lambda_home = (1 - h2h_weight) * lambda_home + h2h_weight * home_h2h["avg_for"]
        lambda_away = (1 - h2h_weight) * lambda_away + h2h_weight * away_h2h["avg_for"]

    return {
        "lambda_home":       round(max(lambda_home, 0.1), 4),
        "lambda_away":       round(max(lambda_away, 0.1), 4),
        "home_attack":       round(home_attack,  3),
        "home_defence":      round(home_defence, 3),
        "away_attack":       round(away_attack,  3),
        "away_defence":      round(away_defence, 3),
        "home_rest_factor":  round(home_rest, 3),
        "away_rest_factor":  round(away_rest, 3),
        "home_injury_factor":round(home_inj,  3),
        "away_injury_factor":round(away_inj,  3),
        "h2h_matches":       home_h2h["matches"],
        "h2h_home_avg":      round(home_h2h["avg_for"], 2),
        "h2h_away_avg":      round(away_h2h["avg_for"], 2),
    }
