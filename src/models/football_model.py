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


def _dixon_coles_correction(home: int, away: int, lam_h: float, lam_a: float, rho: float = -0.04) -> float:
    """
    Dixon-Coles low-score correction factor.
    Adjusts probabilities for 0-0, 1-0, 0-1, 1-1 which Poisson over/under-estimates.
    rho is a small negative correlation parameter. Kept small (-0.04) so that
    the natural Poisson peaks drive the scoreline rather than the correction.
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
    # prob_matrix[h][a]: h=home goals, a=away goals
    # h > a → lower triangle → home win
    # h < a → upper triangle → away win
    win_prob  = float(np.sum(np.tril(prob_matrix, -1)))   # home wins
    draw_prob = float(np.trace(prob_matrix))
    loss_prob = float(np.sum(np.triu(prob_matrix, 1)))    # away wins

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

    # Over/Under: P(total goals > line) from the full probability matrix
    total_goals_dist = {}  # total_goals → probability
    for h in goals_range:
        for a in goals_range:
            t = h + a
            total_goals_dist[t] = total_goals_dist.get(t, 0.0) + prob_matrix[h][a]

    def _over_prob(line: float) -> float:
        return sum(p for t, p in total_goals_dist.items() if t > line)

    ou_lines = {}
    for line in (0.5, 1.5, 2.5, 3.5):
        ou_lines[f"over_{str(line).replace('.','_')}"] = round(_over_prob(line) * 100, 1)

    # Safe bet: highest line where over probability ≥ 65%
    safe_line = None
    safe_prob = None
    for line in (3.5, 2.5, 1.5, 0.5):
        p_over = _over_prob(line)
        if p_over >= 0.65:
            safe_line = line
            safe_prob = round(p_over * 100, 1)
            break
    # Fallback: if none hit 65%, pick under 0.5 direction with highest confidence
    if safe_line is None:
        under_05 = round((1 - _over_prob(0.5)) * 100, 1)
        safe_line = "under_0.5"
        safe_prob = under_05

    # First-half predictions: Poisson is memoryless so HT lambda = full/2
    ht_lambda_home = lambda_home / 2
    ht_lambda_away = lambda_away / 2
    predicted_home_ht = int(np.floor(ht_lambda_home))
    predicted_away_ht = int(np.floor(ht_lambda_away))

    return {
        "predicted_home":    predicted_home,
        "predicted_away":    predicted_away,
        "predicted_home_ht": predicted_home_ht,
        "predicted_away_ht": predicted_away_ht,
        "expected_home":     round(exp_home, 2),
        "expected_away":     round(exp_away, 2),
        "win_probability":   round(win_prob  * 100, 1),
        "draw_probability":  round(draw_prob * 100, 1),
        "loss_probability":  round(loss_prob * 100, 1),
        "confidence":        round(float(prob_matrix[predicted_home][predicted_away]) * 100, 1),
        "top_scorelines":    top5,
        "over_under":        ou_lines,
        "safe_bet":          {"line": safe_line, "type": "over", "probability": safe_prob},
    }
