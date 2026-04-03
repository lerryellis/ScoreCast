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

    sb  = p.get("safe_bet") or {}
    ou  = p.get("over_under") or {}

    record = {
        "sport":             "football",
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
        "safe_bet_line":     str(sb["line"]) if sb.get("line") is not None else None,
        "safe_bet_prob":     sb.get("probability"),
        "over_0_5":          ou.get("over_0_5"),
        "over_1_5":          ou.get("over_1_5"),
        "over_2_5":          ou.get("over_2_5"),
        "over_3_5":          ou.get("over_3_5"),
    }
    try:
        client.table("predictions").upsert(
            record,
            on_conflict="fixture_id,match_date",
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


# ── Basketball save + immediate resolve ────────────────────────────────────────

def _save_basketball_sync(pred: dict) -> None:
    """Save one basketball prediction and immediately resolve if game is final."""
    client = _get_client()
    try:
        client.table("predictions").select("id").limit(1).execute()
    except Exception:
        return

    p   = pred.get("prediction", {})
    sb  = p.get("safe_bet") or {}
    gd  = pred.get("match_date", "") or date.today().isoformat()

    record = {
        "sport":          "basketball",
        "fixture_id":     str(pred.get("game_id", "")),
        "league":         "NBA",
        "league_slug":    "nba",
        "home_team":      pred.get("home_team", ""),
        "away_team":      pred.get("away_team", ""),
        "predicted_home": p.get("predicted_home", 0),
        "predicted_away": p.get("predicted_away", 0),
        "win_prob":       p.get("win_probability"),
        "loss_prob":      p.get("loss_probability"),
        "confidence":     p.get("confidence"),
        "match_date":     gd[:10],
        "safe_bet_line":  str(sb["line"]) if sb.get("line") is not None else None,
        "safe_bet_prob":  sb.get("probability"),
    }
    try:
        resp = client.table("predictions").upsert(
            record, on_conflict="fixture_id,match_date"
        ).execute()
    except Exception as e:
        print(f"[DB basketball save error] {e}")
        return

    # Immediately resolve if final
    if not pred.get("is_final") or pred.get("home_score") is None:
        return

    ah = pred["home_score"]
    aa = pred["away_score"]
    ph = p.get("predicted_home", 0)
    pa = p.get("predicted_away", 0)

    # Fetch the saved row id
    try:
        row = client.table("predictions") \
            .select("id") \
            .eq("fixture_id", str(pred.get("game_id", ""))) \
            .eq("match_date", gd[:10]) \
            .single().execute()
        pred_id = row.data["id"]
    except Exception:
        return

    actual_outcome = "H" if ah > aa else ("A" if aa > ah else "D")
    pred_outcome   = "H" if ph > pa else ("A" if pa > ph else "D")
    sb_correct     = None
    if sb.get("line") is not None:
        try:
            sb_correct = (ah + aa) > float(sb["line"])
        except (ValueError, TypeError):
            pass

    result = {
        "prediction_id":   pred_id,
        "fixture_id":      str(pred.get("game_id", "")),
        "actual_home":     ah,
        "actual_away":     aa,
        "outcome_correct": actual_outcome == pred_outcome,
        "exact_correct":   ah == ph and aa == pa,
        "home_error":      abs(ah - ph),
        "away_error":      abs(aa - pa),
        "safe_bet_correct": sb_correct,
    }
    try:
        client.table("prediction_results").upsert(
            result, on_conflict="prediction_id", ignore_duplicates=True
        ).execute()
    except Exception as e:
        print(f"[DB basketball resolve error] {e}")


async def save_basketball_predictions(preds: list) -> None:
    """Save and immediately resolve finished NBA games."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    tasks = [asyncio.to_thread(_save_basketball_sync, p) for p in preds]
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

        safe_bet_correct = None
        sb_line = pred.get("safe_bet_line")
        if sb_line is not None:
            try:
                safe_bet_correct = (ah + aa) > float(sb_line)
            except (ValueError, TypeError):
                pass

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
            "safe_bet_correct": safe_bet_correct,
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


def _scorecard_sync(sport: str = "football") -> dict:
    client = _get_client()

    # Gracefully handle missing tables (run supabase_schema.sql first)
    try:
        client.table("prediction_results").select("id").limit(1).execute()
    except Exception:
        return {"has_data": False, "total": 0, "needed": MIN_SAMPLE,
                "error": "Tables not created yet — run supabase_schema.sql"}

    rows = (
        client.table("prediction_results")
              .select("*, predictions(sport, league, home_team, away_team, predicted_home, predicted_away, match_date, confidence, safe_bet_line, safe_bet_prob)")
              .execute()
    )
    data = [r for r in (rows.data or []) if (r.get("predictions") or {}).get("sport", "football") == sport]

    total = len(data)
    if total == 0:
        return {"has_data": False, "total": 0, "needed": MIN_SAMPLE}
    if total < MIN_SAMPLE:
        return {"has_data": False, "total": total, "needed": MIN_SAMPLE - total}

    outcome_ok = sum(1 for r in data if r["outcome_correct"])
    exact_ok   = sum(1 for r in data if r["exact_correct"])
    avg_err    = sum(r["home_error"] + r["away_error"] for r in data) / total / 2

    sb_data    = [r for r in data if r.get("safe_bet_correct") is not None]
    sb_hit     = sum(1 for r in sb_data if r["safe_bet_correct"])
    sb_pct     = round(sb_hit / len(sb_data) * 100, 1) if sb_data else None

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
            "date":             p.get("match_date", ""),
            "home_team":        p.get("home_team", ""),
            "away_team":        p.get("away_team", ""),
            "predicted":        f"{p.get('predicted_home')}-{p.get('predicted_away')}",
            "actual":           f"{r['actual_home']}-{r['actual_away']}",
            "outcome_correct":  r["outcome_correct"],
            "exact_correct":    r["exact_correct"],
            "home_error":       r["home_error"],
            "away_error":       r["away_error"],
            "safe_bet_line":    p.get("safe_bet_line"),
            "safe_bet_prob":    p.get("safe_bet_prob"),
            "safe_bet_correct": r.get("safe_bet_correct"),
        })

    return {
        "has_data":    True,
        "total":       total,
        "outcome_pct": round(outcome_ok / total * 100, 1),
        "exact_pct":   round(exact_ok   / total * 100, 1),
        "avg_error":   round(avg_err, 2),
        "safe_bet_pct": sb_pct,
        "safe_bet_total": len(sb_data),
        "leagues":     leagues,
        "recent":      recent,
    }



# ── Bias calibration ───────────────────────────────────────────────────────────

MIN_BIAS_SAMPLE = 15   # don't calibrate until we have this many resolved results

_bias_cache: dict = {}
_bias_cache_date: str = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return round(min(max(v, lo), hi), 4)


def _calibrate_group(items: list) -> dict:
    """
    Compute all learned calibration factors for a group of resolved predictions.
    Returns goal bias, home-advantage factor, rho factor, and avg goals.
    """
    n = len(items)

    # ── Goal bias ──────────────────────────────────────────────────────────
    s_ah = sum(r["actual_home"] for r in items)
    s_aa = sum(r["actual_away"] for r in items)
    p    = lambda r, k: (r.get("predictions") or {}).get(k, 0) or 0
    s_ph = sum(p(r, "predicted_home") for r in items)
    s_pa = sum(p(r, "predicted_away") for r in items)
    home_bias = _clamp(s_ah / s_ph, 0.70, 1.30) if s_ph > 0 else 1.0
    away_bias = _clamp(s_aa / s_pa, 0.70, 1.30) if s_pa > 0 else 1.0

    # ── Home advantage: actual vs predicted home-win rate ──────────────────
    actual_hw  = sum(1 for r in items if r["actual_home"] > r["actual_away"])
    pred_hw    = sum(1 for r in items if p(r, "predicted_home") > p(r, "predicted_away"))
    actual_hw_rate = actual_hw / n
    pred_hw_rate   = pred_hw   / n
    # Adjust HOME_ADVANTAGE_FACTOR proportionally; clamp to [0.80, 1.40]
    home_adv_factor = _clamp(actual_hw_rate / pred_hw_rate, 0.80, 1.40) if pred_hw_rate > 0 else 1.0

    # ── Draw rate: actual vs predicted (tunes Dixon-Coles rho) ─────────────
    actual_draws = sum(1 for r in items if r["actual_home"] == r["actual_away"])
    pred_draws   = sum(1 for r in items if p(r, "predicted_home") == p(r, "predicted_away"))
    actual_draw_rate = actual_draws / n
    pred_draw_rate   = pred_draws   / n
    # rho_factor > 1 → more low-score correction needed; clamp to [0.50, 2.00]
    rho_factor = _clamp(actual_draw_rate / pred_draw_rate, 0.50, 2.00) if pred_draw_rate > 0 else 1.0

    # ── League average goals (actual observed, per team per game) ──────────
    avg_goals = round((s_ah + s_aa) / (2 * n), 4)

    return {
        "home":             home_bias,
        "away":             away_bias,
        "home_adv_factor":  home_adv_factor,
        "rho_factor":       rho_factor,
        "avg_goals":        avg_goals,
        "n":                n,
    }


def _bias_sync() -> dict:
    """
    Compute all model calibration factors from resolved predictions.
    Learns: goal bias, home-advantage factor, rho, league avg goals.
    Returns per-league calibration where N >= 5, plus a global fallback.
    """
    client = _get_client()
    try:
        rows = (
            client.table("prediction_results")
                  .select("actual_home, actual_away, prediction_id, predictions(predicted_home, predicted_away, league, sport)")
                  .execute()
        ).data or []
    except Exception:
        return {"global": {"home": 1.0, "away": 1.0}, "leagues": {}}

    # Filter to football only — basketball scores (100+) would poison goal-based calibration
    rows = [r for r in rows if (r.get("predictions") or {}).get("sport", "football") == "football"]

    if len(rows) < MIN_BIAS_SAMPLE:
        return {"global": {"home": 1.0, "away": 1.0}, "leagues": {}, "n": len(rows)}

    # Global calibration
    global_cal = _calibrate_group(rows)

    # Per-league calibration (only when N >= 5)
    from collections import defaultdict
    by_league: dict = defaultdict(list)
    for r in rows:
        lg = (r.get("predictions") or {}).get("league") or "Unknown"
        by_league[lg].append(r)

    leagues = {lg: _calibrate_group(items)
               for lg, items in by_league.items() if len(items) >= 5}

    return {
        "global":  global_cal,
        "leagues": leagues,
    }


async def get_bias_factors() -> dict:
    """
    Return cached goal-bias factors. Refreshed once per day.
    Falls back to {home: 1.0, away: 1.0} if Supabase unavailable.
    """
    global _bias_cache, _bias_cache_date
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"global": {"home": 1.0, "away": 1.0}, "leagues": {}}
    today = date.today().isoformat()
    if _bias_cache_date != today or not _bias_cache:
        _bias_cache = await asyncio.to_thread(_bias_sync)
        _bias_cache_date = today
    return _bias_cache


async def get_scorecard(sport: str = "football") -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"has_data": False, "total": 0, "needed": MIN_SAMPLE, "error": "Supabase not configured"}
    return await asyncio.to_thread(_scorecard_sync, sport)


# ── Accuracy trend ─────────────────────────────────────────────────────────────

def _trend_sync(sport: str = "football") -> dict:
    """
    Returns daily and 7-day rolling accuracy stats for the performance graph.
    Groups resolved predictions by match_date and computes per-day metrics.
    """
    client = _get_client()
    try:
        rows = (
            client.table("prediction_results")
                  .select("outcome_correct, exact_correct, safe_bet_correct, predictions(sport, match_date, league)")
                  .execute()
        ).data or []
    except Exception:
        return {"daily": [], "rolling7": []}

    rows = [r for r in rows if (r.get("predictions") or {}).get("sport", "football") == sport]

    # Group by date
    by_date: dict = defaultdict(list)
    for r in rows:
        d = (r.get("predictions") or {}).get("match_date") or ""
        if d:
            by_date[d].append(r)

    if not by_date:
        return {"daily": [], "rolling7": []}

    sorted_dates = sorted(by_date.keys())

    daily = []
    for d in sorted_dates:
        items = by_date[d]
        n = len(items)
        outcome_ok = sum(1 for r in items if r["outcome_correct"])
        exact_ok   = sum(1 for r in items if r["exact_correct"])
        sb_items   = [r for r in items if r.get("safe_bet_correct") is not None]
        sb_ok      = sum(1 for r in sb_items if r["safe_bet_correct"])
        daily.append({
            "date":        d,
            "n":           n,
            "outcome_pct": round(outcome_ok / n * 100, 1),
            "exact_pct":   round(exact_ok   / n * 100, 1),
            "safe_bet_pct": round(sb_ok / len(sb_items) * 100, 1) if sb_items else None,
        })

    # 7-day rolling average
    rolling7 = []
    for i, d in enumerate(sorted_dates):
        window = daily[max(0, i - 6): i + 1]
        n_total = sum(w["n"] for w in window)
        if n_total == 0:
            continue
        outcome_avg = round(sum(w["outcome_pct"] * w["n"] for w in window) / n_total, 1)
        exact_avg   = round(sum(w["exact_pct"]   * w["n"] for w in window) / n_total, 1)
        sb_window   = [w for w in window if w["safe_bet_pct"] is not None]
        sb_n        = sum(w["n"] for w in sb_window)
        sb_avg      = round(sum(w["safe_bet_pct"] * w["n"] for w in sb_window) / sb_n, 1) if sb_n else None
        rolling7.append({
            "date":        d,
            "outcome_pct": outcome_avg,
            "exact_pct":   exact_avg,
            "safe_bet_pct": sb_avg,
        })

    return {"daily": daily, "rolling7": rolling7}


async def get_accuracy_trend(sport: str = "football") -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"daily": [], "rolling7": []}
    return await asyncio.to_thread(_trend_sync, sport)
