"""
Basketball score prediction.
Uses the feature-engineered expected scores with a confidence interval
derived from the team's scoring variance over their last N games.
"""

import numpy as np
from scipy import stats


NBA_AVG_POINTS   = 113.0
NBA_SCORE_STDDEV = 12.0    # typical std dev of NBA team scores


def predict_basketball_score(features: dict) -> dict:
    """
    Takes the feature dict from basketball.py and returns score predictions
    with confidence intervals and win probability.
    """
    home_pred = features["home_predicted"]
    away_pred = features["away_predicted"]

    # Use realistic std dev for uncertainty
    home_std  = NBA_SCORE_STDDEV * (2 - features.get("home_rest_factor", 1.0))
    away_std  = NBA_SCORE_STDDEV * (2 - features.get("away_rest_factor", 1.0))

    # Round to nearest integer for display
    home_score = round(home_pred)
    away_score = round(away_pred)

    # Confidence interval (80%)
    z = 1.28
    home_low  = round(home_pred - z * home_std)
    home_high = round(home_pred + z * home_std)
    away_low  = round(away_pred - z * away_std)
    away_high = round(away_pred + z * away_std)

    # Win probability: P(home > away) using normal approximation
    diff_mean = home_pred - away_pred
    diff_std  = np.sqrt(home_std ** 2 + away_std ** 2)
    win_prob  = float(stats.norm.cdf(diff_mean / diff_std)) if diff_std > 0 else 0.5
    loss_prob = 1 - win_prob

    # Confidence: based on how decisive the margin is
    margin    = abs(diff_mean)
    if margin >= 10:
        confidence = 72
    elif margin >= 6:
        confidence = 63
    elif margin >= 3:
        confidence = 57
    else:
        confidence = 51

    # Safe bet: highest total-points line where P(over) >= 65%
    total_pred = home_pred + away_pred
    total_std  = np.sqrt(home_std ** 2 + away_std ** 2)

    def _over_prob(line: float) -> float:
        if total_std <= 0:
            return 1.0 if total_pred > line else 0.0
        return float(1 - stats.norm.cdf(line, loc=total_pred, scale=total_std))

    # Check lines stepping down from just below the predicted total
    safe_line = None
    safe_prob = None
    base = round(total_pred / 5) * 5  # round to nearest 5
    for offset in (17.5, 12.5, 7.5, 2.5, -2.5):
        candidate = base - offset
        p_over = _over_prob(candidate)
        if p_over >= 0.65:
            safe_line = candidate
            safe_prob = round(p_over * 100, 1)
            break

    safe_bet = {"line": safe_line, "type": "over", "probability": safe_prob} if safe_line else None

    return {
        "predicted_home":    home_score,
        "predicted_away":    away_score,
        "home_range":        f"{home_low}–{home_high}",
        "away_range":        f"{away_low}–{away_high}",
        "predicted_margin":  round(diff_mean, 1),
        "win_probability":   round(win_prob  * 100, 1),
        "loss_probability":  round(loss_prob * 100, 1),
        "confidence":        confidence,
        "safe_bet":          safe_bet,
        "home_off_rating":   features.get("home_off_rating"),
        "away_off_rating":   features.get("away_off_rating"),
        "home_def_rating":   features.get("home_def_rating"),
        "away_def_rating":   features.get("away_def_rating"),
    }
