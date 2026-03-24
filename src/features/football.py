"""
Football-specific feature engineering.
Builds the attack/defence strength ratings used by the Dixon-Coles model.

Factors considered:
  1. Form — goals scored/conceded in last 5, last 10, full season
             separately for home games (home team) and away games (away team)
  2. Momentum — points per game in last 5 games
  3. Clean sheet rate — last 10 games
  4. Fixture congestion — games played in last 14 days
  5. H2H — last 5 meetings, venue-specific H2H at this ground
  6. League position — relegation pressure, title motivation
  7. Rest / fatigue — days since last match
"""

import numpy as np
from datetime import date, datetime
from src.features.base import (
    days_since_last_match, rest_factor, h2h_avg_scores, injury_impact_factor
)
from src.config import FOOTBALL_FORM_WINDOW, HOME_ADVANTAGE_FACTOR

LEAGUE_AVG_GOALS = 1.35   # top European league average goals per team per game
TOTAL_TEAMS      = 20     # default league size (PL, La Liga, etc.)


# ── Helper: points per game (momentum) ────────────────────────────────────────
def _ppg(matches: list, n: int = 5) -> float:
    """Points per game from last n matches. W=3, D=1, L=0. Max = 3.0."""
    recent = matches[:n]
    if not recent:
        return 1.5  # neutral
    pts = 0
    for m in recent:
        gf, ga = m.get("goals_for", 0), m.get("goals_ag", 0)
        if gf > ga:
            pts += 3
        elif gf == ga:
            pts += 1
    return pts / len(recent)


# ── Helper: momentum factor from PPG ─────────────────────────────────────────
def _momentum_factor(ppg: float) -> float:
    """
    Converts PPG (0–3) to a lambda multiplier.
    Winning run (PPG ≥ 2.4) → 1.08  |  Losing run (PPG ≤ 0.8) → 0.93
    """
    if ppg >= 2.4:
        return 1.08
    if ppg >= 1.5:
        return 1.03
    if ppg >= 0.8:
        return 1.00
    return 0.93


# ── Helper: clean sheet rate ──────────────────────────────────────────────────
def _clean_sheet_rate(matches: list, n: int = 10) -> float:
    """Fraction of last n games where goals_ag == 0."""
    recent = [m for m in matches[:n] if m.get("goals_ag") is not None]
    if not recent:
        return 0.3
    return sum(1 for m in recent if m["goals_ag"] == 0) / len(recent)


# ── Helper: fixture congestion ────────────────────────────────────────────────
def _congestion_factor(match_dates: list, window_days: int = 14) -> float:
    """
    Penalise teams playing many games in a short window.
    ≥ 4 games in 14 days → 0.94  |  3 games → 0.97  |  ≤ 2 → 1.0
    """
    if not match_dates:
        return 1.0
    cutoff = date.today()
    count = 0
    for d in match_dates:
        try:
            match_date = datetime.fromisoformat(d[:10]).date()
            if 0 <= (cutoff - match_date).days <= window_days:
                count += 1
        except (ValueError, TypeError):
            continue
    if count >= 4:
        return 0.94
    if count == 3:
        return 0.97
    return 1.0


# ── Helper: league position motivation ───────────────────────────────────────
def _position_motivation(rank: int, total: int = TOTAL_TEAMS) -> float:
    """
    Teams fighting relegation or chasing titles get a motivation boost.
    Bottom 3 (relegation): 1.06  |  Top 4 (title/CL chase): 1.03
    Otherwise: 1.0
    """
    if rank == 0:
        return 1.0
    if rank >= total - 2:   # bottom 3
        return 1.06
    if rank <= 4:            # top 4
        return 1.03
    return 1.0


# ── Helper: multi-window form average ────────────────────────────────────────
def _weighted_avg(matches_all: list, key: str, w5: float = 0.5, w10: float = 0.5) -> float:
    """
    Blend last-5 and last-10 averages.
    More recent games carry more weight via w5/w10 split.
    """
    vals5  = [m[key] for m in matches_all[:5]  if m.get(key) is not None]
    vals10 = [m[key] for m in matches_all[:10] if m.get(key) is not None]
    avg5   = float(np.mean(vals5))  if vals5  else None
    avg10  = float(np.mean(vals10)) if vals10 else None
    if avg5 is not None and avg10 is not None:
        return w5 * avg5 + w10 * avg10
    return avg5 or avg10 or LEAGUE_AVG_GOALS


# ── Main feature builder ──────────────────────────────────────────────────────
def build_football_features(
    home_matches:      list,
    away_matches:      list,
    h2h:               list,
    home_team_name:    str,
    away_team_name:    str,
    home_injuries:     list  = None,
    away_injuries:     list  = None,
    home_rank:         int   = 0,
    away_rank:         int   = 0,
    total_teams:       int   = TOTAL_TEAMS,
    home_adv_factor:   float = None,   # learned from resolved predictions
    league_avg_goals:  float = None,   # learned from resolved predictions
) -> dict:
    """
    Returns a feature dict consumed by the football prediction model.
    All match lists are sorted most-recent-first.
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []
    effective_haf  = (HOME_ADVANTAGE_FACTOR * home_adv_factor) if home_adv_factor else HOME_ADVANTAGE_FACTOR
    effective_lag  = league_avg_goals if league_avg_goals else LEAGUE_AVG_GOALS

    # ── 1. Venue-split form ───────────────────────────────────────────────
    MIN_VENUE = 3
    home_at_home = [m for m in home_matches if m.get("is_home")]
    away_at_away = [m for m in away_matches if not m.get("is_home")]

    h_src = home_at_home if len(home_at_home) >= MIN_VENUE else home_matches
    a_src = away_at_away if len(away_at_away) >= MIN_VENUE else away_matches

    # Multi-window blend (50% last-5, 50% last-10) from venue-specific source
    home_avg_scored   = _weighted_avg(h_src, "goals_for")
    home_avg_conceded = _weighted_avg(h_src, "goals_ag")
    away_avg_scored   = _weighted_avg(a_src, "goals_for")
    away_avg_conceded = _weighted_avg(a_src, "goals_ag")

    home_avg_scored   = home_avg_scored   or LEAGUE_AVG_GOALS
    home_avg_conceded = home_avg_conceded or LEAGUE_AVG_GOALS
    away_avg_scored   = away_avg_scored   or LEAGUE_AVG_GOALS
    away_avg_conceded = away_avg_conceded or LEAGUE_AVG_GOALS

    # ── 2. Strength ratings ───────────────────────────────────────────────
    home_attack  = home_avg_scored   / effective_lag
    home_defence = home_avg_conceded / effective_lag
    away_attack  = away_avg_scored   / effective_lag
    away_defence = away_avg_conceded / effective_lag

    # ── 3. Momentum (PPG last 5 — all games, not venue-split) ─────────────
    home_ppg        = _ppg(home_matches, 5)
    away_ppg        = _ppg(away_matches, 5)
    home_momentum   = _momentum_factor(home_ppg)
    away_momentum   = _momentum_factor(away_ppg)

    # ── 4. Clean sheet rate (last 10 — venue-specific) ────────────────────
    home_cs_rate    = _clean_sheet_rate(h_src, 10)
    away_cs_rate    = _clean_sheet_rate(a_src, 10)
    # Higher CS rate → slightly reduce expected goals against
    home_cs_factor  = 1.0 - max(0, home_cs_rate - 0.3) * 0.15
    away_cs_factor  = 1.0 - max(0, away_cs_rate - 0.3) * 0.15

    # ── 5. Fixture congestion ─────────────────────────────────────────────
    home_dates       = [m["date"] for m in home_matches]
    away_dates       = [m["date"] for m in away_matches]
    home_congestion  = _congestion_factor(home_dates)
    away_congestion  = _congestion_factor(away_dates)

    # ── 6. Rest / fatigue ────────────────────────────────────────────────
    home_rest        = rest_factor(days_since_last_match(home_dates))
    away_rest        = rest_factor(days_since_last_match(away_dates))

    # ── 7. Injury impact ─────────────────────────────────────────────────
    home_inj         = injury_impact_factor(home_injuries)
    away_inj         = injury_impact_factor(away_injuries)

    # ── 8. League position motivation ────────────────────────────────────
    home_motivation  = _position_motivation(home_rank, total_teams)
    away_motivation  = _position_motivation(away_rank, total_teams)

    # ── 9. H2H (last 5 overall + venue-specific at this ground) ──────────
    home_h2h         = h2h_avg_scores(h2h, home_team_name)
    away_h2h         = h2h_avg_scores(h2h, away_team_name)

    # Venue-specific H2H: only meetings where home team was actually home
    h2h_at_venue     = [m for m in h2h if m.get("fixture_home_at_home", True)]
    home_h2h_venue   = h2h_avg_scores(h2h_at_venue, home_team_name)
    away_h2h_venue   = h2h_avg_scores(h2h_at_venue, away_team_name)

    # ── 10. Expected goals (λ) ────────────────────────────────────────────
    lambda_home = (
        home_attack * away_defence * effective_lag
        * effective_haf
        * home_rest * home_inj * home_momentum
        * home_congestion * home_motivation
        * away_cs_factor
    )
    lambda_away = (
        away_attack * home_defence * effective_lag
        * away_rest * away_inj * away_momentum
        * away_congestion * away_motivation
        * home_cs_factor
    )

    # ── 11. H2H blend (20% weight when ≥ 3 meetings) ─────────────────────
    if home_h2h["matches"] >= 3:
        # If we also have venue-specific H2H, blend both (60% overall, 40% venue)
        if home_h2h_venue["matches"] >= 2:
            h2h_home_ref = 0.6 * home_h2h["avg_for"] + 0.4 * home_h2h_venue["avg_for"]
            h2h_away_ref = 0.6 * away_h2h["avg_for"] + 0.4 * away_h2h_venue["avg_for"]
        else:
            h2h_home_ref = home_h2h["avg_for"]
            h2h_away_ref = away_h2h["avg_for"]

        h2h_weight  = 0.20
        lambda_home = (1 - h2h_weight) * lambda_home + h2h_weight * h2h_home_ref
        lambda_away = (1 - h2h_weight) * lambda_away + h2h_weight * h2h_away_ref

    return {
        "lambda_home":          round(max(lambda_home, 0.1), 4),
        "lambda_away":          round(max(lambda_away, 0.1), 4),
        "home_attack":          round(home_attack,    3),
        "home_defence":         round(home_defence,   3),
        "away_attack":          round(away_attack,    3),
        "away_defence":         round(away_defence,   3),
        "home_ppg_last5":       round(home_ppg,       2),
        "away_ppg_last5":       round(away_ppg,       2),
        "home_momentum":        round(home_momentum,  3),
        "away_momentum":        round(away_momentum,  3),
        "home_cs_rate":         round(home_cs_rate,   3),
        "away_cs_rate":         round(away_cs_rate,   3),
        "home_rest_factor":     round(home_rest,      3),
        "away_rest_factor":     round(away_rest,      3),
        "home_congestion":      round(home_congestion,3),
        "away_congestion":      round(away_congestion,3),
        "home_motivation":      round(home_motivation,3),
        "away_motivation":      round(away_motivation,3),
        "home_injury_factor":   round(home_inj,       3),
        "away_injury_factor":   round(away_inj,       3),
        "home_rank":            home_rank,
        "away_rank":            away_rank,
        "h2h_matches":          home_h2h["matches"],
        "h2h_venue_matches":    home_h2h_venue["matches"],
        "h2h_home_avg":         round(home_h2h["avg_for"],       2),
        "h2h_away_avg":         round(away_h2h["avg_for"],       2),
        "h2h_venue_home_avg":   round(home_h2h_venue["avg_for"], 2),
        "h2h_venue_away_avg":   round(away_h2h_venue["avg_for"], 2),
    }
