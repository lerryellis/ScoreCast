"""
Elo-style attack/defence ratings — an independent, cross-season/cross-league
signal for team strength, blended equally with the existing form-based
Poisson features (see features/football.py).

Unlike the rolling-form features (last 5/10 games, reset by season/league
boundaries), Elo ratings persist across seasons and carry over when a team
is promoted/relegated (with a markdown/markup — see seed_rating_for_promotion).
That's what lets the model recognise "this team is fundamentally weaker" for
a side like a newly-promoted club, instead of only seeing 5 good games
against weaker opposition (see the SV Elversberg case that motivated this).

Three ratings per team:
  - attack_elo:   how good this team is at scoring
  - defence_elo:  how good this team is at preventing goals
  - midfield_elo: how good this team is at controlling the game (possession,
                   passing) — doesn't feed the Poisson lambda (there's no
                   "goals from possession" formula), but is tracked for the
                   attack/midfield/defence rating display on team cards.
A team's expected goals in a match come from attack_elo (theirs) vs
defence_elo (opponent's) — mirroring how home_attack/away_defence already
combine in the Poisson lambda formula.
"""

BASELINE_ELO = 1500.0   # neutral starting rating — average team, either side
ELO_SCALE    = 400.0    # standard logistic scale (same convention as chess/FIDE)
LEAGUE_AVG_GOALS = 1.35  # kept in sync with features/football.py's constant

# How fast ratings move per match. Lower = more stable/slow-changing,
# higher = more reactive to a single result. Empirically checked (not just
# a guess) via backtest_k_factor() below, replaying goals-only Elo updates
# against the 121 resolved matches that have team IDs on file: MAE was flat
# across K=20-28 and minimized exactly at K=24 (2.2512 vs 2.2517 at K=20/28,
# 2.2596 at K=8, 2.2615 at K=50) — the original hand-picked value happens to
# already be at the empirical optimum for this sample. Re-run the backtest
# as more resolved-with-team-IDs matches accumulate; 121 is a small sample
# and this simplification doesn't include shot/defensive-action stats.
K_FACTOR = 24.0

# Applied once, when a team's league_slug changes between the rating we have
# on file and the league it's playing in now. Reflects that a division
# promotion/relegation is a real change in opponent quality, not something
# the team's own rating should have to "discover" the hard way over several
# matches. ~30 Elo points ≈ a 0.85x/1.15x attack/defence adjustment via the
# logistic formula below — deliberately similar in magnitude to
# CROSS_TIER_ATTACK_DISCOUNT/CROSS_TIER_DEFENCE_INFLATION in
# features/football.py, which does the equivalent job for the raw form
# stats. Promotion is asymmetric from relegation: newly promoted teams
# historically regress harder than relegated teams over-perform.
#
# Still hand-picked, not backtested, for the same reason
# CROSS_TIER_ATTACK_DISCOUNT isn't: there's no persisted record of which
# resolved match involved a promotion/relegation seeding event (it happens
# once, silently, inside get_team_ratings — see seed_rating_for_promotion),
# so there's nothing to replay against yet.
PROMOTION_MARKDOWN_PER_TIER = 40.0   # attack down, defence down (worse) per tier moved up
RELEGATION_MARKUP_PER_TIER  = 25.0   # attack up, defence up (better) per tier moved down

# Shot/defensive-action stat weights for the "composite performance" used in
# rating updates — goals are the primary signal, chances created/prevented
# are a secondary correction so a team that dominated shots but got unlucky
# (or vice versa) isn't rated purely on the scoreline. Deliberately modest
# weights: goals still dominate. Tune once resolved-match volume is large
# enough to fit these properly.
SHOT_ON_TARGET_WEIGHT = 0.10
TOTAL_SHOT_WEIGHT     = 0.03
DEFENSIVE_ACTION_WEIGHT = 0.02   # tackles won + interceptions + clearances + blocked shots + saves

# Midfield rating update — slower-moving than attack/defence (K is lower)
# since possession/passing is noisier match-to-match than goals, and it has
# no direct bearing on the scoreline model, only the display rating.
K_FACTOR_MIDFIELD = 12.0
POSSESSION_WEIGHT  = 0.7
PASS_ACCURACY_WEIGHT = 0.3


# How many resolved matches (with this app's Elo tracking) a team needs
# before Elo carries its full weight in the blend. Below this, Elo's
# influence ramps in linearly from 0 — every team starts at a neutral 1500,
# so blending it in at full strength from game 1 wouldn't add signal, it
# would just dilute the (already working) form-based features toward
# neutral for everyone. Once a team has RAMP_GAMES worth of real results,
# its rating has actually differentiated from the baseline and is worth
# the equal weight it's meant to carry.
RAMP_GAMES = 10


def elo_blend_weight(games_played: int) -> float:
    """0.0 -> 0.5 as games_played goes 0 -> RAMP_GAMES. Caps at 0.5 (equal
    weight with the form-based rating) once a team has enough Elo history."""
    return 0.5 * min(1.0, games_played / RAMP_GAMES)


def elo_to_attack_ratio(attack_elo: float) -> float:
    """Convert an attack Elo into the same multiplicative scale as the
    form-based home_attack/away_attack ratios (1.0 = league average)."""
    return 10 ** ((attack_elo - BASELINE_ELO) / ELO_SCALE)


def elo_to_defence_ratio(defence_elo: float) -> float:
    """Convert a defence Elo into the same multiplicative scale as the
    form-based home_defence/away_defence ratios (1.0 = league average,
    LOWER = better defence — matches the existing convention where
    home_defence = avg_conceded / LEAGUE_AVG_GOALS)."""
    return 10 ** (-(defence_elo - BASELINE_ELO) / ELO_SCALE)


def seed_rating_for_promotion(attack_elo: float, defence_elo: float,
                               old_league_slug: str, new_league_slug: str,
                               midfield_elo: float = None) -> tuple:
    """
    A team's rating is on file under old_league_slug, but they're now playing
    in new_league_slug. If that's a genuine division change (not just a
    fresh season in the same league), apply a one-time markdown/markup so
    the rating reflects the new level of competition immediately, rather
    than needing several matches of results to catch up.

    Division "depth" is inferred from PROMOTION_FEEDER_LEAGUE's chain
    position — e.g. eng.1 is tier 0, eng.2 is tier 1, eng.3 is tier 2.
    Unknown slugs (cups, international) are treated as no-ops.

    Returns (attack_elo, defence_elo, midfield_elo) — midfield_elo is passed
    through unchanged if not given (callers that don't track it yet can omit it).
    """
    if old_league_slug == new_league_slug:
        return attack_elo, defence_elo, midfield_elo

    old_depth = _tier_depth(old_league_slug)
    new_depth = _tier_depth(new_league_slug)
    if old_depth is None or new_depth is None:
        return attack_elo, defence_elo, midfield_elo

    tiers_moved = old_depth - new_depth   # positive = promoted (moved to a lower depth number)
    if tiers_moved > 0:
        # Promoted: attack/midfield down, defence gets worse (defence_elo down) per tier
        delta = PROMOTION_MARKDOWN_PER_TIER * tiers_moved
        new_mid = midfield_elo - delta if midfield_elo is not None else None
        return attack_elo - delta, defence_elo - delta, new_mid
    elif tiers_moved < 0:
        # Relegated: attack/midfield up, defence improves (defence_elo up) per tier
        delta = RELEGATION_MARKUP_PER_TIER * (-tiers_moved)
        new_mid = midfield_elo + delta if midfield_elo is not None else None
        return attack_elo + delta, defence_elo + delta, new_mid
    return attack_elo, defence_elo, midfield_elo


def _tier_depth(league_slug: str) -> int:
    """Returns how many divisions down from the top flight this slug is
    (0 = top flight), by walking PROMOTION_FEEDER_LEAGUE. None if unknown."""
    from src.config import PROMOTION_FEEDER_LEAGUE
    if league_slug in PROMOTION_FEEDER_LEAGUE:
        return 0
    for depth, (top, feeder) in enumerate(_feeder_chain_from_tops()):
        if league_slug == feeder:
            return depth + 1
    return None


def _feeder_chain_from_tops():
    """Flatten PROMOTION_FEEDER_LEAGUE into an ordered walk from each top
    flight down its chain, e.g. eng.1: [(eng.1,eng.2), (eng.2,eng.3), (eng.3,eng.4)]."""
    from src.config import PROMOTION_FEEDER_LEAGUE
    chain = []
    for top in ("eng.1", "esp.1", "ita.1", "ger.1", "fra.1"):
        current = top
        while current in PROMOTION_FEEDER_LEAGUE:
            feeder = PROMOTION_FEEDER_LEAGUE[current]
            chain.append((current, feeder))
            current = feeder
    return chain


def _composite_performance(goals: int, stats: dict = None) -> float:
    """
    "Goals, adjusted for chances" — the actual-performance number an Elo
    update compares against expectation. Falls back to goals-only when shot
    stats aren't available (e.g. lower-division matches ESPN doesn't cover
    in as much detail).
    """
    if not stats:
        return float(goals)
    shots_on_target = float(stats.get("shotsOnTarget", 0) or 0)
    total_shots     = float(stats.get("totalShots", 0) or 0)
    return goals + SHOT_ON_TARGET_WEIGHT * shots_on_target + TOTAL_SHOT_WEIGHT * total_shots


def _defensive_credit(stats: dict = None) -> float:
    """
    Extra credit for a defence that worked hard/well even if the scoreline
    doesn't fully show it (last-ditch tackles, interceptions, clearances,
    blocked shots, keeper saves). Subtracted from the *opponent's* composite
    performance when updating this team's defence rating.
    """
    if not stats:
        return 0.0
    actions = sum(float(stats.get(k, 0) or 0) for k in (
        "effectiveTackles", "interceptions", "effectiveClearance", "blockedShots", "saves"
    ))
    return DEFENSIVE_ACTION_WEIGHT * actions


def update_ratings(home_attack: float, home_defence: float,
                    away_attack: float, away_defence: float,
                    actual_home_goals: int, actual_away_goals: int,
                    home_stats: dict = None, away_stats: dict = None) -> tuple:
    """
    Update both teams' attack/defence Elo ratings after a resolved match.
    Returns (new_home_attack, new_home_defence, new_away_attack, new_away_defence).
    """
    # ── Home attack vs away defence ────────────────────────────────────────
    expected_home_goals = LEAGUE_AVG_GOALS * 10 ** ((home_attack - away_defence) / ELO_SCALE)
    home_performance = _composite_performance(actual_home_goals, home_stats) \
        - _defensive_credit(away_stats)
    home_delta = K_FACTOR * (home_performance - expected_home_goals) / LEAGUE_AVG_GOALS
    new_home_attack  = home_attack + home_delta
    new_away_defence = away_defence - home_delta

    # ── Away attack vs home defence ────────────────────────────────────────
    expected_away_goals = LEAGUE_AVG_GOALS * 10 ** ((away_attack - home_defence) / ELO_SCALE)
    away_performance = _composite_performance(actual_away_goals, away_stats) \
        - _defensive_credit(home_stats)
    away_delta = K_FACTOR * (away_performance - expected_away_goals) / LEAGUE_AVG_GOALS
    new_away_attack  = away_attack + away_delta
    new_home_defence = home_defence - away_delta

    return new_home_attack, new_home_defence, new_away_attack, new_away_defence


def update_midfield_ratings(home_midfield: float, away_midfield: float,
                             home_stats: dict = None, away_stats: dict = None) -> tuple:
    """
    Update both teams' midfield Elo after a resolved match, from possession
    share and pass completion — a zero-sum "who controlled the game" signal,
    updated the same way chess Elo compares actual vs expected score.
    No-ops (returns ratings unchanged) when stats aren't available.
    """
    if not home_stats or not away_stats:
        return home_midfield, away_midfield

    home_possession = float(home_stats.get("possessionPct", 50) or 50) / 100
    home_pass_pct   = float(home_stats.get("passPct", 0.75) or 0.75)
    away_pass_pct   = float(away_stats.get("passPct", 0.75) or 0.75)

    home_performance = POSSESSION_WEIGHT * home_possession + PASS_ACCURACY_WEIGHT * home_pass_pct
    expected_home = 1 / (1 + 10 ** (-(home_midfield - away_midfield) / ELO_SCALE))

    delta = K_FACTOR_MIDFIELD * (home_performance - expected_home)
    return home_midfield + delta, away_midfield - delta


def elo_to_rating_100(elo: float, higher_is_better: bool = True) -> float:
    """
    Convert an Elo rating to a 0-100 display scale, centered at 50 for the
    neutral baseline (1500). Used for the attack/midfield/defence numbers
    shown on team cards — NOT used anywhere in the prediction math (that
    stays in Elo-space / ratio-space, see elo_to_attack_ratio etc).
    """
    direction = 1 if higher_is_better else -1
    return max(0.0, min(100.0, 50.0 + direction * (elo - BASELINE_ELO) / ELO_SCALE * 50.0))


def team_star_rating(attack_elo: float, midfield_elo: float, defence_elo: float) -> float:
    """
    Overall 1-5 star rating (0.5 increments) from the three component
    ratings, for the team-card display. defence_elo is already "higher =
    better" in this system's convention, same as attack/midfield.
    """
    avg_100 = (
        elo_to_rating_100(attack_elo) +
        elo_to_rating_100(midfield_elo) +
        elo_to_rating_100(defence_elo)
    ) / 3.0
    stars = avg_100 / 100.0 * 5.0
    return max(0.5, min(5.0, round(stars * 2) / 2))


async def backtest_k_factor(candidates=(8, 12, 16, 20, 24, 28, 32, 40, 50)) -> dict:
    """
    Empirically check whether K_FACTOR (currently a hand-picked 24.0 — see
    the module-level comment) is actually a good choice, by replaying Elo
    attack/defence updates match-by-match against every resolved match that
    has team IDs on file, for each candidate K, and scoring each candidate
    by how well its resulting PRE-match ratings would have predicted that
    match's actual goals.

    Simplification: goals-only, no shot/defensive-action stats (those
    would need a live ESPN fetch per historical match — out of scope for a
    backtest; the composite-performance weights (SHOT_ON_TARGET_WEIGHT
    etc.) are minor perturbations on top of the goals signal that K_FACTOR
    is primarily sizing anyway). Ratings start fresh at BASELINE_ELO with
    no promotion/relegation seeding, since replaying that accurately would
    need each team's actual tier history — a real limitation given only
    resolved matches with team IDs can be used at all (locked-at-first-save
    predictions from before that column existed can't participate — see
    CLAUDE.md's Team Elo Ratings migration note. Verified live: 121 of 884
    resolved matches, as of this writing).

    Returns {k: mean_absolute_error} — lower is better. Matches are
    replayed in chronological order (oldest first) so ratings accumulate
    the same way they do in production.
    """
    from src.database import _get_client
    client = _get_client()
    rows = (
        client.table("prediction_results")
              .select("actual_home, actual_away, "
                      "predictions(home_team_id, away_team_id, match_date, sport)")
              .execute()
    ).data or []

    matches = [
        r for r in rows
        if (r.get("predictions") or {}).get("sport", "football") == "football"
        and (r.get("predictions") or {}).get("home_team_id")
        and (r.get("predictions") or {}).get("away_team_id")
    ]
    matches.sort(key=lambda r: r["predictions"]["match_date"])

    results = {}
    for k in candidates:
        ratings: dict = {}   # team_id -> {"attack": elo, "defence": elo}
        abs_errors = []
        for r in matches:
            p = r["predictions"]
            h_id, a_id = str(p["home_team_id"]), str(p["away_team_id"])
            hr = ratings.setdefault(h_id, {"attack": BASELINE_ELO, "defence": BASELINE_ELO})
            ar = ratings.setdefault(a_id, {"attack": BASELINE_ELO, "defence": BASELINE_ELO})

            exp_h = LEAGUE_AVG_GOALS * 10 ** ((hr["attack"] - ar["defence"]) / ELO_SCALE)
            exp_a = LEAGUE_AVG_GOALS * 10 ** ((ar["attack"] - hr["defence"]) / ELO_SCALE)
            abs_errors.append(abs(exp_h - r["actual_home"]) + abs(exp_a - r["actual_away"]))

            delta_h = k * (r["actual_home"] - exp_h) / LEAGUE_AVG_GOALS
            delta_a = k * (r["actual_away"] - exp_a) / LEAGUE_AVG_GOALS
            hr["attack"] += delta_h
            ar["defence"] -= delta_h
            ar["attack"] += delta_a
            hr["defence"] -= delta_a

        results[k] = round(sum(abs_errors) / len(abs_errors), 4) if abs_errors else None

    return {"n_matches": len(matches), "mean_abs_error_by_k": results}
