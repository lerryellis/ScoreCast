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
- `fetcher.py` — async HTTP calls to **ESPN's free public API** (primary source for football — fixtures, standings, team schedules, match stats) and `nba_api` package (basketball). `football-data.org` is used as a secondary source for half-time scores during resolution
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

## Data Sources — important, has changed from initial build

**Football's primary source is ESPN's free public API** (`site.api.espn.com/apis/site/v2/sports/soccer/...`), not api-football.com. This matters because:
- api-football.com's **free tier cannot see the current season** — verified live, it errors with "Free plans do not have access to this season, try from 2022 to 2024." It's effectively unusable for live predictions on the free plan.
- ESPN's API needs no key and has no such restriction; it's used for fixtures, standings, team schedules/history, head-to-head, and post-match statistics (shots, tackles, interceptions, possession — used by the Elo system).
- `football-data.org` is still used as a secondary source, specifically for half-time scores during result resolution (`get_football_data_ht_scores`).

Basketball: `nba_api` Python package — free, scrapes NBA.com, no key needed.

## Database — Supabase (Railway is just compute)

**Railway hosts the running app; Supabase (Postgres) is where all persistent data lives.** Schema is in `supabase_schema.sql` — run it in the Supabase SQL Editor when it changes (there is no automated migration path from this codebase; `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements are included for existing deployments since `CREATE TABLE IF NOT EXISTS` doesn't add columns to a table that already exists).

Tables:
- `predictions` — one row per fixture per day (locked at first save; re-runs during/after kickoff never overwrite it — this is what makes the accuracy scorecard meaningful). Includes `home_team_id`/`away_team_id` (added for Elo — previously only team names were stored).
- `prediction_results` — actual scores + correctness flags, written once a match resolves.
- `team_ratings` — Elo ratings (see below).

`src/database.py` also computes **bias calibration** (`get_bias_factors`/`_bias_sync`): a learned home/away goal-bias multiplier and home-advantage/rho factors, computed per-league from resolved prediction errors, refreshed once per day. **Important gotcha**: calibration groups by whatever league name ESPN's scoreboard returns (`"English Premier League"`, `"Spanish LALIGA"`, etc.) which differs from this app's short config keys (`"Premier League"`, `"La Liga"`). `config.normalize_league_name()` / `ESPN_DISPLAY_TO_LEAGUE` bridges this — without it, per-league calibration silently falls back to the global cross-league average for every league. The bias clamp range is `0.60–1.75` (widened from an original `0.70–1.30` that was clamping the learned correction below what the data actually called for, in several leagues it's still pegged at the new ceiling — a sign the real fix is making `LEAGUE_AVG_GOALS` per-league instead of the single global `1.35` constant in `features/football.py`, not yet done).

## Season-boundary / promotion handling (fetcher.py)

Early in a season, or for a newly-promoted team, ESPN's team-schedule endpoint returns nothing under the current league/season — verified live (both no-season-param and the brand-new season number return 0 events right at kickoff). `fetcher._schedule_with_season_fallback()` handles this:
1. Walks back up to 2 prior seasons in the *same* league slug first.
2. If still short on matches (a team that's newly promoted has zero history in the top-flight slug, ever), descends `config.PROMOTION_FEEDER_LEAGUE`'s chain one division at a time (e.g. `eng.1 → eng.2 → eng.3 → eng.4`) until it finds enough matches or the chain runs out.
3. Each match is tagged with a `tier_offset` (0 = same competition, 1+ = pulled from N divisions down) so `features/football.py` can discount it — see next section.

International club friendlies (`fifa.friendly`) are deliberately excluded from `get_intl_team_all_matches`/`get_intl_head_to_head` (`FRIENDLY_COMP_SLUGS`) — squad-rotation friendlies, including ones between the two teams being predicted, aren't representative form/H2H signal.

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

**Migration note**: `team_ratings` and `predictions.home_team_id`/`away_team_id` must exist in Supabase (via `supabase_schema.sql`) for any of this to activate — until then `get_team_ratings()` gracefully degrades to neutral 1500/1500/1500 for every team (nothing breaks, Elo just stays inert).

## GitHub Actions

`.github/workflows/resolve-predictions.yml` — cron every 15 min (13:00–22:00 UTC, covers European kickoffs), hits `POST /api/admin/resolve` on the Railway deployment to mark predictions as won/lost once matches finish. This is the *only* scheduled job; it now also triggers Elo rating updates as a side effect of resolving each match. There is no separate "season sweep" job — the promotion/tier-fallback logic above is fully reactive (resolves itself on request), by design, rather than needing a batch job to run at a season boundary.

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
