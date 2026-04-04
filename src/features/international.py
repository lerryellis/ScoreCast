"""
International football feature engineering — standalone from club football.

Designed for national teams which have fundamentally different characteristics:
  - ~10 games/year vs 50+ for clubs → use ALL available matches, wider windows
  - No venue-split form (neutral venues common in tournaments)
  - No fixture congestion (weeks/months between games)
  - Momentum is less meaningful — squads change every window
  - H2H is more important (repeated matchups in qualifiers)
"""

import numpy as np
from src.features.base import h2h_avg_scores

INTL_AVG_GOALS       = 1.30   # international matches average fewer goals
INTL_HOME_ADVANTAGE  = 1.10   # weaker home advantage than club football


def _ppg_intl(matches: list) -> float:
    """Points per game from ALL available matches. W=3, D=1, L=0."""
    if not matches:
        return 1.5
    pts = 0
    for m in matches:
        gf, ga = m.get("goals_for", 0), m.get("goals_ag", 0)
        if gf > ga:
            pts += 3
        elif gf == ga:
            pts += 1
    return pts / len(matches)


def _momentum_intl(ppg: float) -> float:
    """
    Muted momentum for international — less signal from sparse games.
    Strong form (PPG >= 2.2) → 1.05 | Weak (PPG <= 0.8) → 0.96
    """
    if ppg >= 2.2:
        return 1.05
    if ppg >= 1.4:
        return 1.02
    if ppg >= 0.8:
        return 1.00
    return 0.96


def _clean_sheet_rate(matches: list) -> float:
    """Fraction of all matches where goals_ag == 0."""
    relevant = [m for m in matches if m.get("goals_ag") is not None]
    if not relevant:
        return 0.25
    return sum(1 for m in relevant if m["goals_ag"] == 0) / len(relevant)


def build_international_features(
    home_matches:   list,
    away_matches:   list,
    h2h:            list,
    home_team_name: str,
    away_team_name: str,
) -> dict:
    """
    Returns a feature dict for international football.
    Uses ALL available matches (not windowed) due to small sample sizes.
    """
    # ── 1. Scoring averages (all matches, no venue split) ────────────────
    def _avg(matches, key, fallback):
        vals = [m[key] for m in matches if m.get(key) is not None]
        return float(np.mean(vals)) if vals else fallback

    home_avg_scored   = _avg(home_matches, "goals_for", INTL_AVG_GOALS)
    home_avg_conceded = _avg(home_matches, "goals_ag",  INTL_AVG_GOALS)
    away_avg_scored   = _avg(away_matches, "goals_for", INTL_AVG_GOALS)
    away_avg_conceded = _avg(away_matches, "goals_ag",  INTL_AVG_GOALS)

    # ── 2. Strength ratings ──────────────────────────────────────────────
    home_attack  = home_avg_scored   / INTL_AVG_GOALS
    home_defence = home_avg_conceded / INTL_AVG_GOALS
    away_attack  = away_avg_scored   / INTL_AVG_GOALS
    away_defence = away_avg_conceded / INTL_AVG_GOALS

    # ── 3. Momentum (all games — muted factor) ──────────────────────────
    home_ppg       = _ppg_intl(home_matches)
    away_ppg       = _ppg_intl(away_matches)
    home_momentum  = _momentum_intl(home_ppg)
    away_momentum  = _momentum_intl(away_ppg)

    # ── 4. Clean sheet rate ──────────────────────────────────────────────
    home_cs_rate   = _clean_sheet_rate(home_matches)
    away_cs_rate   = _clean_sheet_rate(away_matches)
    home_cs_factor = 1.0 - max(0, home_cs_rate - 0.25) * 0.10
    away_cs_factor = 1.0 - max(0, away_cs_rate - 0.25) * 0.10

    # ── 5. H2H (heavier weight — 30% for international) ─────────────────
    home_h2h = h2h_avg_scores(h2h, home_team_name)
    away_h2h = h2h_avg_scores(h2h, away_team_name)

    # ── 6. Sample size confidence ────────────────────────────────────────
    # Shrink ratings toward 1.0 when few games available
    n_home = len(home_matches)
    n_away = len(away_matches)
    # With 10+ games, full confidence; with 3, ~50% shrinkage toward mean
    home_conf = min(n_home / 10, 1.0)
    away_conf = min(n_away / 10, 1.0)

    home_attack  = home_conf * home_attack  + (1 - home_conf) * 1.0
    home_defence = home_conf * home_defence + (1 - home_conf) * 1.0
    away_attack  = away_conf * away_attack  + (1 - away_conf) * 1.0
    away_defence = away_conf * away_defence + (1 - away_conf) * 1.0

    # ── 7. Expected goals (λ) ────────────────────────────────────────────
    lambda_home = (
        home_attack * away_defence * INTL_AVG_GOALS
        * INTL_HOME_ADVANTAGE
        * home_momentum * away_cs_factor
    )
    lambda_away = (
        away_attack * home_defence * INTL_AVG_GOALS
        * away_momentum * home_cs_factor
    )

    # ── 8. H2H blend (30% weight when >= 2 meetings) ────────────────────
    if home_h2h["matches"] >= 2:
        h2h_weight  = 0.30
        lambda_home = (1 - h2h_weight) * lambda_home + h2h_weight * home_h2h["avg_for"]
        lambda_away = (1 - h2h_weight) * lambda_away + h2h_weight * away_h2h["avg_for"]

    return {
        "lambda_home":       round(max(lambda_home, 0.1), 4),
        "lambda_away":       round(max(lambda_away, 0.1), 4),
        "home_attack":       round(home_attack,    3),
        "home_defence":      round(home_defence,   3),
        "away_attack":       round(away_attack,    3),
        "away_defence":      round(away_defence,   3),
        "home_ppg":          round(home_ppg,       2),
        "away_ppg":          round(away_ppg,       2),
        "home_momentum":     round(home_momentum,  3),
        "away_momentum":     round(away_momentum,  3),
        "home_cs_rate":      round(home_cs_rate,   3),
        "away_cs_rate":      round(away_cs_rate,   3),
        "home_sample_size":  n_home,
        "away_sample_size":  n_away,
        "home_confidence":   round(home_conf, 2),
        "away_confidence":   round(away_conf, 2),
        "h2h_matches":       home_h2h["matches"],
        "h2h_home_avg":      round(home_h2h["avg_for"], 2),
        "h2h_away_avg":      round(away_h2h["avg_for"], 2),
    }
