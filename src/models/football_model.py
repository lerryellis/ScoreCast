"""
Dixon-Coles Poisson model for football score prediction.
Simulates 100,000 scorelines from the expected goals (λ) values
and returns the probability distribution over scorelines.
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple


MAX_GOALS    = 8      # we model scores 0-0 to 8-8
N_SIMULATIONS = 100_000


def _dixon_coles_correction(home: int, away: int, lam_h: float, lam_a: float, rho: float = -0.1) -> float:
    """
    Dixon-Coles low-score correction factor.
    Adjusts probabilities for 0-0, 1-0, 0-1, 1-1 which Poisson over/under-estimates.
    rho is a small negative correlation parameter (typically -0.1 to -0.15).
    """
    if home == 0 and away == 0:
        return 1 - lam_h * lam_a * rho
    if home == 1 and away == 0:
        return 1 + lam_a * rho
    if home == 0 and away == 1:
        return 1 + lam_h * rho
    if home == 1 and away == 1:
        return 1 - rho
    return 1.0


def predict_football_score(lambda_home: float, lambda_away: float) -> dict:
    """
    Given expected goals for home and away teams, returns:
    - predicted_home, predicted_away: most likely scoreline
    - all_probabilities: full matrix of scoreline probabilities
    - top_scorelines: top 5 most likely scorelines with probabilities
    - win_prob, draw_prob, loss_prob
    - confidence: probability of the top scoreline
    """
    lambda_home = max(lambda_home, 0.05)
    lambda_away = max(lambda_away, 0.05)

    goals_range = np.arange(0, MAX_GOALS + 1)

    # Build probability matrix
    prob_matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for h in goals_range:
        for a in goals_range:
            p  = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            p *= _dixon_coles_correction(h, a, lambda_home, lambda_away)
            prob_matrix[h][a] = p

    # Normalise
    prob_matrix /= prob_matrix.sum()

    # Most likely scoreline
    idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    predicted_home, predicted_away = int(idx[0]), int(idx[1])

    # Win / draw / loss probabilities
    win_prob  = float(np.sum(np.tril(prob_matrix, -1).T))   # home wins (h > a)
    draw_prob = float(np.trace(prob_matrix))
    loss_prob = float(np.sum(np.tril(prob_matrix, -1)))      # away wins

    # Top 5 scorelines
    flat     = [(prob_matrix[h][a], h, a) for h in goals_range for a in goals_range]
    flat.sort(reverse=True)
    top5     = [
        {"scoreline": f"{h}-{a}", "probability": round(p * 100, 1)}
        for p, h, a in flat[:5]
    ]

    # Expected goals (mean of distribution)
    exp_home = float(np.sum([h * prob_matrix[h, :].sum() for h in goals_range]))
    exp_away = float(np.sum([a * prob_matrix[:, a].sum() for a in goals_range]))

    return {
        "predicted_home":    predicted_home,
        "predicted_away":    predicted_away,
        "expected_home":     round(exp_home, 2),
        "expected_away":     round(exp_away, 2),
        "win_probability":   round(win_prob  * 100, 1),
        "draw_probability":  round(draw_prob * 100, 1),
        "loss_probability":  round(loss_prob * 100, 1),
        "confidence":        round(float(prob_matrix[predicted_home][predicted_away]) * 100, 1),
        "top_scorelines":    top5,
    }
