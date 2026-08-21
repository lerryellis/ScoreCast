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
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

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
