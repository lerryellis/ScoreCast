-- Run this in your Supabase SQL Editor (Dashboard → SQL Editor → New query)

CREATE TABLE IF NOT EXISTS predictions (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  fixture_id    TEXT        NOT NULL,
  league        TEXT,
  league_slug   TEXT,
  home_team     TEXT        NOT NULL,
  away_team     TEXT        NOT NULL,
  home_team_id  TEXT,
  away_team_id  TEXT,
  predicted_home     INTEGER NOT NULL,
  predicted_away     INTEGER NOT NULL,
  predicted_home_ht  INTEGER,
  predicted_away_ht  INTEGER,
  lambda_home   FLOAT,
  lambda_away   FLOAT,
  win_prob      FLOAT,
  draw_prob     FLOAT,
  loss_prob     FLOAT,
  confidence    FLOAT,
  match_date    DATE        NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  -- Raw match-context features (not just the Poisson model's own outputs) —
  -- lets the XGBoost correction layer learn genuine team/matchup corrections
  -- instead of only "the base model is biased when its own numbers look
  -- like X". NULL on any row saved before this was added (locked at first
  -- save, by design) — models/ml_model.py substitutes neutral defaults for
  -- those, same pattern as every other locked-history gap this app has.
  home_attack       FLOAT,
  home_defence      FLOAT,
  away_attack       FLOAT,
  away_defence      FLOAT,
  home_rank         INTEGER,
  away_rank         INTEGER,
  home_rest_factor  FLOAT,
  away_rest_factor  FLOAT,
  h2h_matches       INTEGER,
  home_tier_gap     FLOAT,
  away_tier_gap     FLOAT
);

-- Existing deployments: CREATE TABLE IF NOT EXISTS above won't add columns
-- to a table that already exists.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_attack FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_defence FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_attack FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_defence FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_rank INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_rank INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_rest_factor FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_rest_factor FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS h2h_matches INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_tier_gap FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_tier_gap FLOAT;

-- Prevent duplicate saves for the same fixture on the same day
CREATE UNIQUE INDEX IF NOT EXISTS predictions_fixture_date_uidx
  ON predictions (fixture_id, match_date);

CREATE TABLE IF NOT EXISTS prediction_results (
  id             UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  prediction_id  UUID REFERENCES predictions(id) ON DELETE CASCADE,
  fixture_id     TEXT    NOT NULL,
  actual_home    INTEGER NOT NULL,
  actual_away    INTEGER NOT NULL,
  actual_home_ht INTEGER,
  actual_away_ht INTEGER,
  outcome_correct   BOOLEAN NOT NULL,
  exact_correct     BOOLEAN NOT NULL,
  home_error        INTEGER NOT NULL,
  away_error        INTEGER NOT NULL,
  ht_exact_correct  BOOLEAN,
  resolved_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS results_prediction_uidx
  ON prediction_results (prediction_id);

-- Existing deployments: CREATE TABLE IF NOT EXISTS above won't add columns
-- to a table that already exists, so add them explicitly here too.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS home_team_id TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS away_team_id TEXT;

-- Team Elo ratings — attack/defence strength per team, independent of and
-- blended with the form-based Poisson features. Persists across seasons and
-- carries over (with a promotion/relegation adjustment — see
-- src/models/elo.py) when a team changes divisions. One row per team.
CREATE TABLE IF NOT EXISTS team_ratings (
  team_id       TEXT PRIMARY KEY,
  team_name     TEXT,
  league_slug   TEXT,
  sport         TEXT NOT NULL DEFAULT 'football',
  attack_elo    FLOAT NOT NULL DEFAULT 1500,
  defence_elo   FLOAT NOT NULL DEFAULT 1500,
  midfield_elo  FLOAT NOT NULL DEFAULT 1500,
  games_played  INTEGER NOT NULL DEFAULT 0,
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Existing deployments (table already created without this column):
ALTER TABLE team_ratings ADD COLUMN IF NOT EXISTS midfield_elo FLOAT NOT NULL DEFAULT 1500;

-- Persistent backing store for src/cache.py's ESPN fetch cache. The
-- in-memory layer (per-process, wiped on every Railway restart/redeploy)
-- stays the fast first check; this survives restarts, so a fresh process
-- boot doesn't have to re-fetch from ESPN for anything still within its
-- TTL from before the restart. One row per cache key.
CREATE TABLE IF NOT EXISTS espn_cache (
  cache_key   TEXT PRIMARY KEY,
  data        JSONB NOT NULL,
  expires_at  TIMESTAMPTZ NOT NULL,
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Fast lookup for cleanup/expiry checks
CREATE INDEX IF NOT EXISTS espn_cache_expires_idx ON espn_cache (expires_at);
