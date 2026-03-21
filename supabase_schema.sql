-- Run this in your Supabase SQL Editor (Dashboard → SQL Editor → New query)

CREATE TABLE IF NOT EXISTS predictions (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  fixture_id    TEXT        NOT NULL,
  league        TEXT,
  league_slug   TEXT,
  home_team     TEXT        NOT NULL,
  away_team     TEXT        NOT NULL,
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
