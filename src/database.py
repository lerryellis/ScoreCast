"""
Supabase database layer.
Saves predictions when they are computed, resolves them after games finish,
and provides scorecard aggregates for the accuracy tab.
"""

import asyncio
from datetime import date, timedelta
from collections import defaultdict
from typing import Optional

from src.config import SUPABASE_URL, SUPABASE_KEY

# ── Client (lazy singleton) ────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Save prediction ────────────────────────────────────────────────────────────

def _save_prediction_sync(pred: dict) -> None:
    """Insert one prediction row; ignore if already saved (unique constraint)."""
    client = _get_client()
    # Silently skip if table doesn't exist yet
    try:
        client.table("predictions").select("id").limit(1).execute()
    except Exception:
        return  # Tables not created yet — run supabase_schema.sql
    p  = pred.get("prediction", {})
    f  = pred.get("features", {})
    mt = pred.get("match_time", "") or ""

    record = {
        "fixture_id":        str(pred.get("fixture_id", "")),
        "league":            pred.get("league", ""),
        "league_slug":       pred.get("league_slug", ""),
        "home_team":         pred.get("home_team", ""),
        "away_team":         pred.get("away_team", ""),
        "predicted_home":    p.get("predicted_home", 0),
        "predicted_away":    p.get("predicted_away", 0),
        "predicted_home_ht": p.get("predicted_home_ht"),
        "predicted_away_ht": p.get("predicted_away_ht"),
        "lambda_home":       f.get("lambda_home"),
        "lambda_away":       f.get("lambda_away"),
        "win_prob":          p.get("win_probability"),
        "draw_prob":         p.get("draw_probability"),
        "loss_prob":         p.get("loss_probability"),
        "confidence":        p.get("confidence"),
        "match_date":        mt[:10] if mt else date.today().isoformat(),
    }
    try:
        client.table("predictions").upsert(
            record,
            on_conflict="fixture_id,match_date",
            ignore_duplicates=True,
        ).execute()
    except Exception as e:
        print(f"[DB save error] {e}")


async def save_prediction(pred: dict) -> None:
    """Fire-and-forget: save one football prediction in a thread."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    await asyncio.to_thread(_save_prediction_sync, pred)


async def save_predictions(preds: list) -> None:
    """Save a batch of predictions without blocking the response."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    tasks = [asyncio.create_task(save_prediction(p)) for p in preds]
    await asyncio.gather(*tasks, return_exceptions=True)


# ── Resolve predictions ────────────────────────────────────────────────────────

def _resolve_sync(unresolved: list, fd_by_date: dict) -> int:
    """Match unresolved predictions to actual scores and write results."""
    from src.fetcher import match_ht_to_fixture
    client = _get_client()
    count  = 0

    for pred in unresolved:
        match_date = pred["match_date"]
        fd_matches = fd_by_date.get(match_date, [])
        fd = match_ht_to_fixture(fd_matches, pred["home_team"], pred["away_team"])
        if not fd or fd.get("home_ft") is None:
            continue

        ah = fd["home_ft"]
        aa = fd["away_ft"]
        ph = pred["predicted_home"]
        pa = pred["predicted_away"]

        actual_outcome = "H" if ah > aa else ("A" if aa > ah else "D")
        pred_outcome   = "H" if ph > pa else ("A" if pa > ph else "D")

        ht_correct = None
        if fd.get("home_ht") is not None and pred.get("predicted_home_ht") is not None:
            ht_correct = (
                fd["home_ht"] == pred["predicted_home_ht"] and
                fd["away_ht"] == pred["predicted_away_ht"]
            )

        result = {
            "prediction_id":  pred["id"],
            "fixture_id":     pred["fixture_id"],
            "actual_home":    ah,
            "actual_away":    aa,
            "actual_home_ht": fd.get("home_ht"),
            "actual_away_ht": fd.get("away_ht"),
            "outcome_correct": actual_outcome == pred_outcome,
            "exact_correct":   ah == ph and aa == pa,
            "home_error":      abs(ah - ph),
            "away_error":      abs(aa - pa),
            "ht_exact_correct": ht_correct,
        }
        try:
            client.table("prediction_results").upsert(
                result,
                on_conflict="prediction_id",
                ignore_duplicates=True,
            ).execute()
            count += 1
        except Exception as e:
            print(f"[DB resolve error] {e}")

    return count


async def resolve_predictions() -> int:
    """
    Fetch unresolved past predictions, look up actual scores from football-data.org,
    and write results. Returns the number of predictions resolved.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return 0

    from src.fetcher import get_football_data_ht_scores

    client  = _get_client()
    today   = date.today().isoformat()

    # Predictions for past dates
    rows = await asyncio.to_thread(
        lambda: client.table("predictions")
                      .select("*")
                      .lt("match_date", today)
                      .execute()
    )
    if not rows.data:
        return 0

    # Which ones already have results?
    done = await asyncio.to_thread(
        lambda: client.table("prediction_results")
                      .select("prediction_id")
                      .execute()
    )
    resolved_ids = {r["prediction_id"] for r in (done.data or [])}
    unresolved   = [p for p in rows.data if p["id"] not in resolved_ids]

    if not unresolved:
        return 0

    # Group by date, fetch FD scores per date in parallel
    by_date = defaultdict(list)
    for p in unresolved:
        by_date[p["match_date"]].append(p)

    fd_results = await asyncio.gather(
        *[get_football_data_ht_scores(d) for d in by_date]
    )
    fd_by_date = {d: fd for d, fd in zip(by_date.keys(), fd_results)}

    return await asyncio.to_thread(_resolve_sync, unresolved, fd_by_date)


# ── Scorecard ──────────────────────────────────────────────────────────────────

MIN_SAMPLE = 30   # don't show stats until we have this many resolved predictions


def _scorecard_sync() -> dict:
    client = _get_client()

    # Gracefully handle missing tables (run supabase_schema.sql first)
    try:
        client.table("prediction_results").select("id").limit(1).execute()
    except Exception:
        return {"has_data": False, "total": 0, "needed": MIN_SAMPLE,
                "error": "Tables not created yet — run supabase_schema.sql"}

    rows = (
        client.table("prediction_results")
              .select("*, predictions(league, home_team, away_team, predicted_home, predicted_away, match_date, confidence)")
              .execute()
    )
    data = rows.data or []

    total = len(data)
    if total == 0:
        return {"has_data": False, "total": 0, "needed": MIN_SAMPLE}
    if total < MIN_SAMPLE:
        return {"has_data": False, "total": total, "needed": MIN_SAMPLE - total}

    outcome_ok = sum(1 for r in data if r["outcome_correct"])
    exact_ok   = sum(1 for r in data if r["exact_correct"])
    avg_err    = sum(r["home_error"] + r["away_error"] for r in data) / total / 2

    # Per-league
    by_league: dict[str, list] = defaultdict(list)
    for r in data:
        lg = (r.get("predictions") or {}).get("league") or "Unknown"
        by_league[lg].append(r)

    leagues = []
    for lg, items in sorted(by_league.items(), key=lambda x: -len(x[1])):
        n = len(items)
        leagues.append({
            "league":       lg,
            "games":        n,
            "outcome_pct":  round(sum(1 for r in items if r["outcome_correct"]) / n * 100, 1),
            "exact_pct":    round(sum(1 for r in items if r["exact_correct"])   / n * 100, 1),
            "avg_error":    round(sum(r["home_error"] + r["away_error"] for r in items) / n / 2, 2),
        })

    # Recent 25
    recent_raw = sorted(
        data,
        key=lambda r: (r.get("predictions") or {}).get("match_date") or "",
        reverse=True,
    )[:25]

    recent = []
    for r in recent_raw:
        p = r.get("predictions") or {}
        recent.append({
            "date":            p.get("match_date", ""),
            "home_team":       p.get("home_team", ""),
            "away_team":       p.get("away_team", ""),
            "predicted":       f"{p.get('predicted_home')}-{p.get('predicted_away')}",
            "actual":          f"{r['actual_home']}-{r['actual_away']}",
            "outcome_correct": r["outcome_correct"],
            "exact_correct":   r["exact_correct"],
            "home_error":      r["home_error"],
            "away_error":      r["away_error"],
        })

    return {
        "has_data":    True,
        "total":       total,
        "outcome_pct": round(outcome_ok / total * 100, 1),
        "exact_pct":   round(exact_ok   / total * 100, 1),
        "avg_error":   round(avg_err, 2),
        "leagues":     leagues,
        "recent":      recent,
    }


async def get_scorecard() -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"has_data": False, "total": 0, "needed": MIN_SAMPLE, "error": "Supabase not configured"}
    return await asyncio.to_thread(_scorecard_sync)
