"""
Shared feature engineering logic used by both sports.
"""

import numpy as np
from datetime import date, datetime
from typing import Optional


def rolling_average(values: list, window: int) -> float:
    """Mean of last `window` values. Returns 0 if list is empty."""
    if not values:
        return 0.0
    trimmed = values[-window:]
    return float(np.mean(trimmed))


def days_since_last_match(match_dates: list[str]) -> int:
    """Number of days since the most recent match date string (ISO format)."""
    if not match_dates:
        return 7   # default: assume a week's rest
    latest = max(datetime.fromisoformat(d[:10]) for d in match_dates)
    return (date.today() - latest.date()).days


def rest_factor(days_rest: int) -> float:
    """
    A multiplier reflecting fatigue or freshness.
    < 3 days  → fatigued  (0.92)
    3–5 days  → normal    (1.00)
    6–10 days → fresh     (1.04)
    > 10 days → rusty     (0.97)
    """
    if days_rest < 3:
        return 0.92
    if days_rest <= 5:
        return 1.00
    if days_rest <= 10:
        return 1.04
    return 0.97


def h2h_avg_scores(h2h_records: list, team_name: str) -> dict:
    """
    Given a list of h2h records, compute average goals/points for
    `team_name` and against them.
    Returns {"avg_for": float, "avg_ag": float, "matches": int}
    """
    scored = []
    conceded = []

    for match in h2h_records:
        if match.get("home_team") == team_name:
            scored.append(match.get("home_goals", match.get("home_score", 0)))
            conceded.append(match.get("away_goals", match.get("away_score", 0)))
        elif match.get("away_team") == team_name:
            scored.append(match.get("away_goals", match.get("away_score", 0)))
            conceded.append(match.get("home_goals", match.get("home_score", 0)))

    if not scored:
        return {"avg_for": 0.0, "avg_ag": 0.0, "matches": 0}

    return {
        "avg_for":  float(np.mean(scored)),
        "avg_ag":   float(np.mean(conceded)),
        "matches":  len(scored),
    }


def injury_impact_factor(injuries: list, sport: str = "football") -> float:
    """
    Reduce expected performance based on missing players.
    Each key player missing reduces output by a small multiplier.
    """
    if not injuries:
        return 1.0
    # Each injury/suspension reduces factor by 2.5%
    reduction = min(len(injuries) * 0.025, 0.15)   # cap at 15%
    return round(1.0 - reduction, 4)
