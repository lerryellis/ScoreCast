# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env          # then add API keys
uvicorn src.api:app --reload --port 8000   # dev server → localhost:8000
```

## Architecture

ScoreCast is a sports score prediction app covering football/soccer and NBA basketball.

**Backend (FastAPI — `src/`)**
- `api.py` — FastAPI app, serves `index.html` + exposes `/api/predictions/football` and `/api/predictions/basketball`, plus admin endpoints (`/api/admin/resolve`, `/api/admin/train`)
- `predictor.py` — orchestrates: fixture → features → model → prediction dict. Also fetches Elo ratings and attaches `home_ratings`/`away_ratings` (see Team Elo Ratings below)
- `fetcher.py` — async HTTP calls to **ESPN's free public API** (primary and, as of this writing, only source for football — fixtures, standings, team schedules, match stats, and result resolution; see Data Sources below for why `football-data.org` was dropped) and `nba_api` package (basketball)
- `database.py` — Supabase persistence layer: saves predictions, resolves them against actual results, computes bias calibration, and maintains Elo ratings (see below)
- `config.py` — API keys, league IDs/slugs, model constants, league-name normalization, promotion/feeder-league chains

**Feature Engineering (`src/features/`)**
- `base.py` — shared helpers: rolling averages, rest factor, H2H, injury impact
- `football.py` — attack/defence strength ratings (form-based, cross-tier-discounted, Elo-blended) → λ_home, λ_away (expected goals)
- `basketball.py` — offensive/defensive ratings, pace, fatigue → predicted points

**Models (`src/models/`)**
- `football_model.py` — Dixon-Coles Poisson: builds 9×9 scoreline probability matrix, returns most likely scoreline + top 5 + win/draw/loss %
- `basketball_model.py` — score prediction with confidence intervals from scoring variance
- `ml_model.py` — XGBoost correction layer on top of the Poisson lambdas. **Football uses `objective="count:poisson"`, basketball uses `objective="reg:squarederror"`** — using Poisson for basketball's ~100-160 point scores overflows (verified: MAE went to ~1e21 before the fix)
- `elo.py` — Elo-style attack/midfield/defence ratings per team (see Team Elo Ratings below)

**Frontend (`index.html`)**
- Pure HTML/CSS/JS — no build step
- Dark/light sports-themed UI, tabs for Football / International / Basketball / Accuracy / Performance
- Calls `/api/predictions/*` endpoints, renders match cards; clicking a card opens a modal (`openMatchModal` → `renderModalContent`) with scores, probability bars, scorelines, form/H2H, and team star ratings

## Model Internals — algorithms, formulas, considerations

Three layers run in sequence for every football prediction: **feature pipeline → Dixon-Coles Poisson → XGBoost correction**, then Elo blends into the feature pipeline as a fourth input. This section documents the actual math, not just the names.

### 1. Feature pipeline → λ (expected goals) — `features/football.py`

Everything reduces to two numbers, `lambda_home` and `lambda_away` (expected goals for each side), built as a chain of multiplicative factors around a league-average baseline:

```python
lambda_home = (
    home_attack * away_defence * LEAGUE_AVG_GOALS   # LEAGUE_AVG_GOALS = 1.35
    * HOME_ADVANTAGE_FACTOR
    * home_rest * home_inj * home_momentum
    * home_congestion * home_motivation
    * home_rank_quality
    * away_cs_factor
)
# lambda_away mirrors this with attack/defence and home/away swapped
```

Each factor is a ratio centered at **1.0 = league average**, so the whole expression reads as "how many goals above/below league-average, adjusted by every consideration." Where each factor comes from:

- **`home_attack` / `home_defence`** — `avg_goals_scored / LEAGUE_AVG_GOALS` and `avg_goals_conceded / LEAGUE_AVG_GOALS`, blended 50/50 from last-5 and last-10 games (`_weighted_avg`), venue-split (home games only for the home team, if ≥3 available). This is the "raw form" signal.
- **Cross-tier discount** — if that form data was pulled from a feeder league (`tier_offset > 0`, see Season-boundary handling below), `home_attack *= 0.85**tier_offset` and `home_defence *= 1.15**tier_offset` (`CROSS_TIER_ATTACK_DISCOUNT`/`CROSS_TIER_DEFENCE_INFLATION`) — a promoted team's dominant lower-division numbers get discounted before they reach λ.
- **Elo blend** — `home_attack = (1-w)*home_attack + w*elo_to_attack_ratio(attack_elo)` where `w` ramps 0→0.5 with games played (see Elo section). Applied *after* the cross-tier discount, on the same 1.0-centered scale.
- **`home_momentum`** — points-per-game over last 5 (`_ppg`) mapped through a step function: PPG≥2.4 → 1.08, ≥1.5 → 1.03, ≥0.8 → 1.00, else 0.93.
- **`home_cs_factor`** (applied to the *opponent's* λ) — clean-sheet rate over last 10 reduces the opponent's expected goals: `1.0 - max(0, cs_rate - 0.3) * 0.15`.
- **`home_congestion`** — games in the last 14 days across *all* competitions (cups count): ≥4 → 0.94, 3 → 0.97, ≤2 → 1.0.
- **`home_rest`** — from `days_since_last_match`, via `features/base.py::rest_factor`.
- **`home_motivation`** — table position: bottom-3 (relegation fight) → 1.06, top-4 (title/CL chase) → 1.03, else 1.0. Zeroed out (rank treated as unknown) below `MIN_GAMES_FOR_RANK` (3) games played, since ESPN's standings list every team 1..N before kickoff with `games_played=0` — not a real position.
- **`home_rank_quality`** — the *relative* gap between the two teams' table positions: `1.0 + (away_rank - home_rank)/total_teams * 0.10`, clamped to `[0.90, 1.10]`.
- **H2H blend** (applied after λ is otherwise complete) — if ≥3 head-to-head meetings exist, blend 20% weight toward the average goals in those meetings (`h2h_weight = 0.20`); if ≥2 of those were at this specific venue, that venue-specific average gets 40% of the H2H portion (`0.6*overall + 0.4*venue`).

**Considerations / known limitations:**
- `LEAGUE_AVG_GOALS = 1.35` is a *single global constant* used for every league and country. This is the still-open root cause behind Serie A/Ligue 1/EPL bias hitting the calibration clamp ceiling (see Bias calibration above) — the fix is making it per-league, not done yet.
- All the "1.0 = average" factors are multiplied together, so their effects compound — a team that's simultaneously in great form, well-rested, and motivated can stack multiple >1.0 factors into a large swing. No factor is normalized against the others.
- Rank/motivation/momentum constants (0.93–1.10 range) are hand-picked, not fit from data.

### 2. Dixon-Coles Poisson scoreline model — `models/football_model.py`

Takes `(lambda_home, lambda_away)` and returns a full scoreline probability distribution. **This is an exact analytical calculation, not a Monte Carlo simulation** — despite the module docstring/`N_SIMULATIONS = 100_000` constant claiming otherwise (dead/stale comment from an earlier implementation; worth cleaning up if touching this file).

```python
for h in 0..8:            # MAX_GOALS = 8, i.e. a 9×9 matrix
  for a in 0..8:
    p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)   # independent Poisson
    p *= dixon_coles_correction(h, a, rho)                          # low-score adjustment
    prob_matrix[h][a] = p
prob_matrix /= prob_matrix.sum()   # renormalize (truncating at 8 loses a little mass)
```

- **Independent Poisson base**: standard assumption that home and away goals are independent Poisson processes with rates λ_home/λ_away. This is the well-known weakness Dixon-Coles (1997) addresses — real football has a slight negative correlation (very low-scoring games, especially 0-0/1-0/0-1/1-1, are *more* common than independence predicts, because a team already leading tends to sit back).
- **`_dixon_coles_correction(h, a, rho)`**: multiplies those four low-score cells by `1 - λ_h·λ_a·ρ` (0-0), `1 + λ_a·ρ` (1-0), `1 + λ_h·ρ` (0-1), `1 - ρ` (1-1); every other cell is unadjusted (`1.0`). `rho` is small and negative (`effective_rho = max(-0.15, min(0, -0.04 * rho_factor))`, where `rho_factor` comes from bias calibration) — deliberately kept small so the natural Poisson shape drives the result and this only nudges the low-score cells.
- **Most likely scoreline**: `argmax` over the full matrix.
- **Win/draw/loss**: the matrix is literally triangular-split — `np.tril(matrix, -1).sum()` = home-win probability (all cells where home goals > away goals), `np.trace(matrix)` = draw probability (the diagonal, h==a), `np.triu(matrix, 1).sum()` = away-win. Clean because the matrix rows/columns are goals-scored, so "home wins" is exactly "below the diagonal."
- **Top-5 scorelines**: flatten and sort by probability, no surprises.
- **Over/Under lines**: group matrix cells by `h+a` into a total-goals distribution, then sum tail probability past each line (0.5/1.5/2.5/3.5).
- **"Safe bet"**: highest over-line where P(over) ≥ 65%, checked from 3.5 down to 0.5; falls back to "under 0.5" if nothing clears that bar.
- **Half-time scores**: exploits the Poisson memoryless property — `ht_lambda = full_lambda / 2`, floored. This assumes goals are scored at a uniform rate through the match, which is a simplification (real football is scored more heavily in the second half on average), but is a reasonable default absent half-specific data.

### 3. XGBoost ML correction layer — `models/ml_model.py`

Runs *after* the Poisson model, as a learned correction on top of it — not a replacement:

```
fixture → features → Poisson(λ) → base_pred  →  XGBoost(base_pred's own outputs as features)  →  corrected (λ_home, λ_away)  →  Poisson re-run with corrected λ  →  final prediction
```

- **Football features fed to XGBoost**: `lambda_home, lambda_away, win_prob, draw_prob, loss_prob, confidence, predicted_home, predicted_away, over_0_5..over_3_5, league_enc` — i.e. it's learning a correction *from the Poisson model's own output*, treating the Poisson model as a feature generator rather than re-deriving from raw match data. `league_enc` is an integer league ID assigned in training-order (`{league: index}`, 0 = unknown/unseen league).
- **Objective differs by sport, and this matters**: football uses `objective="count:poisson"` (correct for small counts, 0-5ish goals). Basketball uses `objective="reg:squarederror"` — using Poisson's log-link at basketball's 70-160 point scale overflowed and produced a cross-validated MAE of ~1e21 before this was split out; squared-error is the right objective for that continuous scale.
- **Training**: `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)`, one regressor for home goals and one for away, trained on resolved `prediction_results` joined with their original `predictions` row. Needs ≥15 samples or training refuses (`{"error": "Insufficient data"}`).
- **Inference sanity clamps**: football output outside `[0.05, 8.0]` λ, or basketball outside `[70, 160]` points, is **rejected outright** (`predict()` returns `None`, caller falls back to the un-corrected Poisson/base prediction) — guards against the log-link extrapolating wildly on inputs outside the training distribution.
- **Model persistence**: pickled `SportsMLModel` instances at `models/football_ml_latest.pkl`/`models/basketball_ml_latest.pkl`, loaded once as a module-level singleton (`get_football_ml()`/`get_basketball_ml()`).

**Considerations:** the feature set is *entirely derived from the Poisson model's own output* plus a league ID — there's no raw match-context feature (no form/rest/injury signal directly), so this layer can only learn "the Poisson model tends to be off in this direction when its own outputs look like X," not anything about the match itself independently. A league added after the last training run always encodes to 0 (unknown) until retrained.

### 4. Elo ratings — `models/elo.py`

See the "Team Elo Ratings" section below for the product-level why. This is the formula-level how. Three independent ratings per team — `attack_elo`, `defence_elo`, `midfield_elo` — each baseline `1500.0`, logistic scale `400.0` (same convention as chess/FIDE Elo, i.e. a 400-point gap = the stronger side is a 10:1 favorite on that dimension).

**Converting Elo to the model's ratio scale** (used to blend into `home_attack`/`home_defence`):
```python
elo_to_attack_ratio(elo)  = 10 ** ((elo - 1500) / 400)     # >1500 → ratio > 1.0
elo_to_defence_ratio(elo) = 10 ** (-(elo - 1500) / 400)    # >1500 → ratio < 1.0 (better defence = fewer relative goals conceded)
```

**Post-match update** (`update_ratings`) — standard Elo actual-vs-expected, but the "actual" side is a *composite performance* number, not just the raw goal count:
```python
expected_home_goals = LEAGUE_AVG_GOALS * 10 ** ((home_attack_elo - away_defence_elo) / 400)
home_performance    = goals_home + 0.10*shots_on_target_home + 0.03*total_shots_home
                       - 0.02*(away's tackles+interceptions+clearances+blocked_shots+saves)
delta                = K_FACTOR * (home_performance - expected_home_goals) / LEAGUE_AVG_GOALS   # K_FACTOR = 24.0
new_home_attack_elo  = home_attack_elo + delta
new_away_defence_elo = away_defence_elo - delta   # symmetric: the away side's defence rating absorbs the same swing
```
Mirrored for away-attack vs home-defence. The shot/defensive-action weights are a "goals, adjusted for chances" signal — a team that dominated shots but got unlucky isn't marked down as hard as the scoreline alone would suggest, and a defence that made a lot of last-ditch tackles/interceptions/clearances even in a loss gets partial credit. Falls back to goals-only (`_composite_performance` returns `goals` unchanged) when ESPN doesn't have shot stats for that match.

**Midfield update** (`update_midfield_ratings`) — separate, lower `K_FACTOR_MIDFIELD = 12.0` (possession is noisier match-to-match than goals), uses the standard chess-Elo expected-score formula since possession is naturally zero-sum:
```python
home_performance = 0.7*possession_share_home + 0.3*pass_completion_home
expected_home     = 1 / (1 + 10 ** (-(home_mid - away_mid) / 400))
delta             = K_FACTOR_MIDFIELD * (home_performance - expected_home)
```

**Promotion/relegation seeding** (`seed_rating_for_promotion`) — a one-time markdown/markup applied when a team's rating (on file under one `league_slug`) is looked up under a different one. Tier depth comes from walking `PROMOTION_FEEDER_LEAGUE`'s chain (`eng.1`=0, `eng.2`=1, `eng.3`=2, `eng.4`=3, similarly for esp/ita/ger/fra):
```python
tiers_moved = old_depth - new_depth
if promoted (tiers_moved > 0): attack -= 40*tiers_moved; defence -= 40*tiers_moved   # PROMOTION_MARKDOWN_PER_TIER
if relegated (tiers_moved < 0): attack += 25*|tiers_moved|; defence += 25*|tiers_moved|   # RELEGATION_MARKUP_PER_TIER
```
Asymmetric by design — promoted teams historically regress harder than relegated teams overperform.

**Blend weight ramp** (`elo_blend_weight`) — `0.5 * min(1.0, games_played / 10)`. Documented at length in the Team Elo Ratings section below; the short version is a flat 50/50 blend against a team's still-neutral 1500 rating actively hurts predictions until real match history accumulates, so the weight ramps in instead of being constant.

**0-100 display scale + stars** (`elo_to_rating_100`, `team_star_rating`) — pure display transform, doesn't feed back into any prediction math: `50 + (elo-1500)/400 * 50`, clamped `[0,100]`; stars = average of the three 0-100 numbers, scaled to `/5` and rounded to the nearest 0.5.

**Considerations:** `K_FACTOR` (24), `K_FACTOR_MIDFIELD` (12), the shot/defensive-action weights (0.10/0.03/0.02), and the promotion/relegation markdown/markup (40/25) are all **hand-picked starting points, not fit from data** — there isn't yet enough resolved-match volume through this system to calibrate them properly. Revisit once a season's worth of resolved matches has accumulated.

## Data Sources — important, has changed from initial build

**Football's primary source is ESPN's free public API** (`site.api.espn.com/apis/site/v2/sports/soccer/...`), not api-football.com. This matters because:
- api-football.com's **free tier cannot see the current season** — verified live, it errors with "Free plans do not have access to this season, try from 2022 to 2024." It's effectively unusable for live predictions on the free plan.
- ESPN's API needs no key and has no such restriction; it's used for fixtures, standings, team schedules/history, head-to-head, and post-match statistics (shots, tackles, interceptions, possession — used by the Elo system).
- `football-data.org` is **no longer used** — its account was found disabled (`403 "Your account has been disabled"`, verified live across every competition, not a rate limit) which had silently stopped `resolve_predictions()` from resolving anything at all for some time (blocking the accuracy scorecard, bias calibration, and Elo alike, since none of them ever saw a real outcome). Replaced with `fetcher.get_espn_fixture_result()` — one ESPN `summary?event=` call per fixture returns actual FT/HT score + stats + team IDs, fetched directly by `fixture_id` (more precise than football-data.org's old fuzzy team-name matching ever was, too).

Basketball: `nba_api` Python package — free, scrapes NBA.com, no key needed.

**Known issue: ESPN occasionally 403s, most likely a request-velocity throttle, not a static IP ban** (investigated live 2026-08-27). First observed as 403s from Railway's egress IP while the exact same request succeeded from a dev machine — but later that same dev machine also started getting 403s after enough of its own rapid requests, then recovered after a pause. Reads as a general rate/velocity throttle on ESPN's CDN that any sufficiently bursty client can trip and that resets after a cooldown, with Railway's shared egress IP more exposed to it since it carries traffic from many tenants at once (not something this app's own request pattern uniquely causes). **Do not try fixing this by spoofing a browser User-Agent** — already tried and verified live to make things *worse*: the same request that succeeded with httpx's default UA started failing with a Chrome UA attached, on the same machine, same moment. ESPN's WAF almost certainly fingerprints the mismatch between a claimed browser UA and httpx's actual TLS/HTTP handshake as more suspicious than an honest non-browser client.

**The actual fix is `src/cache.py`** — an adaptive-TTL, request-coalescing cache in front of every ESPN-calling function on the page-load/refresh hot path (fixtures, standings, team form, H2H, calendar dates, and the NBA equivalents). Three things matter here: (1) TTL varies by data volatility via a per-function `ttl_fn` — today's live fixtures get 30s, a team's last-5-games gets 15 min, a fully-past season's schedule gets 6h, so live scores stay fresh without needlessly re-fetching static data; (2) concurrent callers for the same key while a fetch is in flight share that one call (verified: 10 concurrent identical requests → 1 real ESPN call), so N users refreshing at once doesn't fan out into N ESPN hits; (3) **backed by Supabase (`espn_cache` table)**, not just in-memory — the in-memory layer is wiped on every Railway restart/redeploy, which happens often during active development, so a persistent layer sits behind it: on an in-memory miss, check Supabase before ever calling ESPN. In-memory stays the fast path (no network at all on a hit); Supabase is only consulted on a miss, so this costs nothing on the hot path. Verified with a mocked restart simulation (wipe in-memory only): the next call served from Supabase with zero additional ESPN calls. **Important interaction**: the cached functions raise on failure rather than swallowing it — an earlier version of this fix caught ESPN errors *inside* the fetch functions and returned `[]`/`{}` to avoid 500ing the page, which then got **cached as a false "no fixtures" answer**, silently freezing a transient error for the full TTL. Fixed by moving failure-tolerance to the *callers* that fan out over multiple slugs/leagues (`asyncio.gather(..., return_exceptions=True)` in `get_all_football_predictions` and the fixture-dates endpoints) instead of inside the cached functions themselves — the right layer for "one slug failing shouldn't break everything" without corrupting what gets cached. If throttling starts happening often enough to matter beyond what caching absorbs, revisit request volume from `MULTI_SLUG_LEAGUES` leagues (2-4x the calls of a normal league) before anything else.

**Migration note**: `espn_cache` must exist in Supabase (via `supabase_schema.sql`) for the persistent layer to activate — until then, `get_espn_cache_entry`/`set_espn_cache_entry` gracefully degrade to a no-op (one warning logged per process, not one per fetch — every cached call touches this table, so without the once-only guard it floods the logs identically to the pre-migration `team_ratings` situation). Caching still works in-memory-only until the migration runs; nothing breaks.

**Fallback providers evaluated and rejected (2026-08-27)** — a request came in to add API-Football, KickoffAPI, football-data.org, and TheSportsDB as fallbacks for when ESPN is rate-limited. None are actually usable right now:
- **API-Football**: already integrated (legacy) but confirmed broken on free tier — see above, can't see the current season at all.
- **football-data.org**: already integrated but confirmed disabled account (see above) — same failure mode as ESPN, wouldn't help.
- **TheSportsDB** (`get_thesportsdb_day`): partially integrated, but its response has **team names only, no ESPN team IDs**, no live in-play status, no venue/logos. Every prediction (form, Elo, H2H) is keyed by ESPN team ID, so this data can't feed the prediction engine — only a plain "today's fixtures/results" list, which is already its sole current use (`/api/predictions/day`).
- **KickoffAPI**: not integrated, no credentials/docs available to build against.

Decision: rely on the caching layer rather than multi-provider fallback. Revisit only if the user obtains working credentials for one of these (a new football-data.org account, an upgraded API-Football plan, or a KickoffAPI key).

## Database — Supabase (Railway is just compute)

**Railway hosts the running app; Supabase (Postgres) is where all persistent data lives.** Schema is in `supabase_schema.sql` — run it in the Supabase SQL Editor when it changes (there is no automated migration path from this codebase; `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements are included for existing deployments since `CREATE TABLE IF NOT EXISTS` doesn't add columns to a table that already exists).

Tables:
- `predictions` — one row per fixture per day (locked at first save; re-runs during/after kickoff never overwrite it — this is what makes the accuracy scorecard meaningful). Includes `home_team_id`/`away_team_id` (added for Elo — previously only team names were stored).
- `prediction_results` — actual scores + correctness flags, written once a match resolves.
- `team_ratings` — Elo ratings (see below).

`src/database.py` also computes **bias calibration** (`get_bias_factors`/`_bias_sync`): a learned home/away goal-bias multiplier and home-advantage/rho factors, computed per-league from resolved prediction errors, refreshed once per day. **Important gotcha**: calibration groups by whatever league name ESPN's scoreboard returns (`"English Premier League"`, `"Spanish LALIGA"`, etc.) which differs from this app's short config keys (`"Premier League"`, `"La Liga"`). `config.normalize_league_name()` / `ESPN_DISPLAY_TO_LEAGUE` bridges this — without it, per-league calibration silently falls back to the global cross-league average for every league. The bias clamp range is `0.60–1.75` (widened from an original `0.70–1.30` that was clamping the learned correction below what the data actually called for, in several leagues it's still pegged at the new ceiling — a sign the real fix is making `LEAGUE_AVG_GOALS` per-league instead of the single global `1.35` constant in `features/football.py`, not yet done).

**Related, separate bug also found and fixed**: `_save_prediction_sync` hardcoded `"sport": "football"` for every saved prediction — including international ones, saved through the same function. Verified live: every stored international-fixture row had `sport="football"` despite `league="International Friendly"`. Two consequences: `get_bias_factors(sport="international")` could never find a matching row (silently useless), and football's own calibration pool was quietly contaminated with international-friendly error patterns mixed in. Now reads `pred.get("sport")`. Pre-existing mislabeled rows aren't retroactively fixable (locked at first save).

## Season-boundary / promotion handling (fetcher.py)

Early in a season, or for a newly-promoted team, ESPN's team-schedule endpoint returns nothing under the current league/season — verified live (both no-season-param and the brand-new season number return 0 events right at kickoff). `fetcher._schedule_with_season_fallback()` handles this:
1. Walks back up to 2 prior seasons in the *same* league slug first.
2. If still short on matches (a team that's newly promoted has zero history in the top-flight slug, ever), descends `config.PROMOTION_FEEDER_LEAGUE`'s chain one division at a time (e.g. `eng.1 → eng.2 → eng.3 → eng.4`) until it finds enough matches or the chain runs out.
3. Each match is tagged with a `tier_offset` (0 = same competition, 1+ = pulled from N divisions down) so `features/football.py` can discount it — see next section.

International club friendlies (`fifa.friendly`) are deliberately excluded from `get_intl_team_all_matches`/`get_intl_head_to_head` (`FRIENDLY_COMP_SLUGS`) — squad-rotation friendlies, including ones between the two teams being predicted, aren't representative form/H2H signal.

**NBA has the same season-boundary bug, fixed the same way, but with an inverted season-numbering convention.** `get_espn_nba_team_games()` queries with no `season` param, which returns the *upcoming* season's fixtures (all `STATUS_SCHEDULED`, zero completed) once ESPN's "current season" default flips ahead of opening night — verified live for the off-season/preseason window. Fix walks back seasons the same way, **but**: ESPN's NBA `season` param is the year the season *ends* (`season=2026` = the 2025-26 season) — the opposite of the soccer endpoints, where `season=2025` means the 2025-26 season (year it *starts*). Verified live for both conventions before writing the fix. `date.today().year` works directly as the first fallback candidate under NBA's convention, walking back up to 3 seasons. No promotion/relegation concept in the NBA, so no feeder-league chain needed here — just the season walkback.

## UEFA competitions: qualifying rounds + women's competitions (config.py)

ESPN runs each UEFA club competition's qualifying/play-off rounds under a **separate slug** from its league phase — verified live (2026-08-26/27) that `UEFA.CHAMPIONS`/`UEFA.EUROPA`/`UEFA.EUROPA.CONF` alone had zero fixtures on days when real play-off second legs were being played (Fenerbahçe vs Lyon, Real Madrid vs Ajax, KuPS Kuopio vs Shamrock Rovers, etc.) — those were under `UEFA.CHAMPIONS_QUAL`/`UEFA.EUROPA_QUAL`/`UEFA.EUROPA.CONF_QUAL` instead. Same split for the Women's Champions League (`uefa.wchampions` / `uefa.wchampions_qual`) — no Women's Europa or Conference League exists (checked several slug variants live, all 400).

`config.MULTI_SLUG_LEAGUES` maps a single dropdown league name to a list of ESPN slugs; `get_all_football_predictions()` fetches and merges fixtures from every slug in the list instead of just one. Applies to:
- **"Champions League"** → men's + women's, league phase + qualifying (4 slugs) — one dropdown entry rather than separate Men/Women entries, which would make the league dropdown grow indefinitely.
- **"Europa League"** / **"Conference League"** → league phase + qualifying (2 slugs each), men's only.

Every fixture already carries its own correct `league_slug`/`league` display name from whichever ESPN endpoint it was merged in from, so no downstream prediction code needed changes for this. Women's fixtures are tagged `is_women` (via `config.WOMENS_SLUGS` membership) on the prediction dict, and `index.html` renders a small pink "W" badge next to the league name (match card, modal, and Safe Bets row) wherever that's true — this is how the UI tells the merged men's/women's fixtures apart without a separate dropdown entry per gender.

`/api/football/fixture-dates` (the calendar's green-highlight source) merges across the same slug list — it was originally only checking the primary slug, so the calendar silently missed every day that only had qualifying-round or women's fixtures.

## Cross-tier form discount (features/football.py)

A promoted team's feeder-league scoring stats overstate how it'll perform against stronger opposition (verified live: SV Elversberg's 2. Bundesliga form made them look like a title-chasing attack against Bayer Leverkusen — 70.5% predicted win probability before this fix). `CROSS_TIER_ATTACK_DISCOUNT` (0.85) / `CROSS_TIER_DEFENCE_INFLATION` (1.15) are applied per tier of `tier_offset` gap before the ratings feed the Poisson lambda.

Also in `predictor.py`: ESPN's standings endpoint lists every team at rank 1..N before a ball is kicked (`games_played=0` for everyone — alphabetical/seeded placeholder order, not a real position). `MIN_GAMES_FOR_RANK` (3) guards against treating that as a real quality signal.

## Team Elo Ratings (src/models/elo.py + database.py)

A cross-season, cross-league strength prior, added because form-only features (reset every season/league boundary) can't tell "this team is fundamentally weaker," only "this team's last 5 games looked good against whoever it played." Three ratings per team, baseline 1500, logistic scale 400 (chess/FIDE convention):
- `attack_elo` / `defence_elo` — feed directly into the Poisson lambda (blended with the form-based attack/defence ratios).
- `midfield_elo` — possession/pass-completion based; display-only, doesn't affect the scoreline model (no goals-from-possession formula).

Updated after every match `resolve_predictions()` resolves, using goals **plus** shot/defensive-action stats pulled from ESPN's free `summary?event=` endpoint (`get_espn_match_stats` — shots on target, tackles, interceptions, clearances, saves; verified live, no key needed). Promotion/relegation applies a one-time markdown/markup (`seed_rating_for_promotion`) using the same `PROMOTION_FEEDER_LEAGUE` chain, rather than making the team's rating "discover" the level change over several matches.

**Blend weight ramps, it isn't flat 50/50** — `elo_blend_weight()` scales 0 → 0.5 over `RAMP_GAMES` (10) resolved matches. This was a deliberate deviation from a literal always-equal-weight blend: verified live that blending a neutral 1500 baseline at full strength for a team with zero rating history actively diluted the (already-correct) form-based signal toward neutral for *every* team, not just cold-start ones. Once a team has enough resolved matches for its rating to have actually differentiated from baseline, it carries the full equal weight.

Display: `predictor._team_ratings_display()` converts the raw Elo into `{attack, midfield, defence}` on a 0-100 scale (`elo_to_rating_100`, centered at 50) plus an overall 1-5 star rating (`team_star_rating`, 0.5 increments) — shown in `index.html`'s match modal under each team name (click-to-reveal), tagged "provisional" below `RAMP_GAMES` resolved matches. Basketball has no Elo tracking (football-only for now).

**Migration note**: `team_ratings` and `predictions.home_team_id`/`away_team_id` must exist in Supabase (via `supabase_schema.sql`) for any of this to activate — until then `get_team_ratings()` gracefully degrades to neutral 1500/1500/1500 for every team (nothing breaks, Elo just stays inert). As of 2026-08-21 this migration has been run and the resolve pipeline's football-data.org outage (see Data Sources) has been fixed — 692/788 previously-stuck predictions were backfilled in one pass, 125 teams already have real (non-baseline) ratings. `home_team_id`/`away_team_id` are `NULL` forever on any prediction row saved before that column existed (locked at first save, by design) — `resolve_predictions()` recovers the IDs from the same ESPN fetch as a fallback, so that backlog isn't permanently stranded even without a backfill migration.

## GitHub Actions

`.github/workflows/resolve-predictions.yml` — cron every 15 min (13:00–22:00 UTC, covers European kickoffs), hits `POST /api/admin/resolve` on the Railway deployment to mark predictions as won/lost once matches finish. This is the *only* scheduled job; it now also triggers Elo rating updates as a side effect of resolving each match. There is no separate "season sweep" job — the promotion/tier-fallback logic above is fully reactive (resolves itself on request), by design, rather than needing a batch job to run at a season boundary.

## In-process background loops (api.py `startup`)

Two `asyncio.create_task()` loops start when the app boots (not GitHub Actions — these run inside the same long-lived process, so they stop if the app restarts and resume fresh on the next boot):
- **`_auto_resolve_loop()`** — resolves yesterday's predictions once daily at 00:05 UTC, then retrains the ML models.
- **`_cache_warm_loop()`** — proactively refreshes every tracked ESPN slug's fixtures (+ NBA scoreboard) every 5 minutes, so a visitor's page load hits an already-warm `src/cache.py` cache instead of triggering a live ESPN fetch during their own request. Deliberately not on the cache's own 30s "live day" TTL — running that aggressively 24/7 regardless of actual traffic would itself become a large constant source of ESPN load, which is the exact problem the cache exists to avoid. A real visitor refreshing during a live match still gets sub-30s freshness from the normal reactive cache on top of this baseline.

## Key Environment Variables
```
API_FOOTBALL_KEY    # from api-football.com — legacy/mostly unused now, see Data Sources above
SUPABASE_URL        # Postgres persistence — predictions, calibration, Elo ratings
SUPABASE_KEY
ADMIN_KEY           # gates /api/admin/* endpoints
PORT                # default 8000
```

## Deployment
- Railway: `Procfile` already configured (`uvicorn src.api:app --host 0.0.0.0 --port $PORT`). Deploys from `main` — merge/push there for changes to go live.
- Vercel: not applicable (Python backend) — serve `index.html` from Railway directly.
- Supabase schema changes require manually running `supabase_schema.sql` in the Supabase SQL Editor — there's no automated migration step in the deploy pipeline.
