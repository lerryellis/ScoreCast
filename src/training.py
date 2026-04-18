"""
ML training pipeline.

Fetches resolved predictions from Supabase, trains separate XGBoost models
for football and basketball, saves them to disk, and hot-swaps the in-memory
singletons so predictions improve immediately without a server restart.
"""

import asyncio
from src.config import SUPABASE_URL, SUPABASE_KEY
from src.models.ml_model import (
    SportsMLModel,
    FOOTBALL_MODEL_PATH,
    BASKETBALL_MODEL_PATH,
    set_football_ml,
    set_basketball_ml,
)

MIN_TRAIN_SAMPLES = 30


async def fetch_resolved(sport: str) -> list:
    """Pull all resolved predictions from Supabase for a given sport."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def _fetch():
        return (
            client.table("prediction_results")
                  .select(
                      "actual_home, actual_away, "
                      "predictions("
                      "  sport, league, "
                      "  lambda_home, lambda_away, "
                      "  predicted_home, predicted_away, "
                      "  win_prob, draw_prob, loss_prob, "
                      "  confidence, "
                      "  over_0_5, over_1_5, over_2_5, over_3_5"
                      ")"
                  )
                  .execute()
        ).data or []

    rows = await asyncio.to_thread(_fetch)
    # Filter to the requested sport, dropping rows with missing prediction data
    return [
        r for r in rows
        if (r.get("predictions") or {}).get("sport") == sport
        and r.get("actual_home") is not None
        and r.get("actual_away") is not None
    ]


async def train_model(sport: str) -> dict:
    """Fetch data, train, save to disk, hot-swap in-memory singleton."""
    rows = await fetch_resolved(sport)

    if len(rows) < MIN_TRAIN_SAMPLES:
        msg = f"Not enough resolved data for {sport}: {len(rows)} < {MIN_TRAIN_SAMPLES}"
        print(f"[ML training] {msg}")
        return {"error": msg, "n": len(rows)}

    model = SportsMLModel(sport)
    metrics = model.train(rows)

    if "error" in metrics:
        print(f"[ML training] {sport} failed: {metrics['error']}")
        return metrics

    path = FOOTBALL_MODEL_PATH if sport == "football" else BASKETBALL_MODEL_PATH
    model.save(path)

    if sport == "football":
        set_football_ml(model)
    else:
        set_basketball_ml(model)

    print(
        f"[ML training] {sport} model trained — "
        f"n={metrics['n_samples']}, "
        f"home_MAE={metrics['home_mae_cv']}, "
        f"away_MAE={metrics['away_mae_cv']}, "
        f"outcome_acc={metrics.get('outcome_acc')}%"
    )
    return metrics


async def train_all() -> dict:
    """Train models for all sports in parallel. Returns combined metrics."""
    football, basketball = await asyncio.gather(
        train_model("football"),
        train_model("basketball"),
    )
    return {"football": football, "basketball": basketball}
