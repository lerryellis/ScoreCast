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

    return {
        "predicted_home":    home_score,
        "predicted_away":    away_score,
        "home_range":        f"{home_low}–{home_high}",
        "away_range":        f"{away_low}–{away_high}",
        "predicted_margin":  round(diff_mean, 1),
        "win_probability":   round(win_prob  * 100, 1),
        "loss_probability":  round(loss_prob * 100, 1),
        "confidence":        confidence,
        "home_off_rating":   features.get("home_off_rating"),
        "away_off_rating":   features.get("away_off_rating"),
        "home_def_rating":   features.get("home_def_rating"),
        "away_def_rating":   features.get("away_def_rating"),
    }
