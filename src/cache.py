"""
Generic in-memory TTL cache for ESPN fetches, with request coalescing.

Why this exists: every page load / refresh-button click re-fetches fixtures,
standings, team form, H2H, and calendar data from ESPN — with dozens of
concurrent users (or just one user refreshing repeatedly), that's a lot of
avoidable traffic to a free, unauthenticated, rate-limited public API (see
CLAUDE.md's "Known issue: ESPN occasionally 403s" note — this is the actual
fix for that, not a header trick).

Two things this needs to get right:
1. Adaptive TTL — a live match's score needs to look fresh within a minute
   or so; a team's "last 5 games" barely changes within a whole day. One
   flat TTL for everything is either too slow for live data or needlessly
   re-fetches static data. Each cached function gets its own TTL policy
   (see espn_cache's ttl_fn param) instead of a single global number.
2. Request coalescing — if 20 users refresh at the same moment right after
   a cache entry expires, they should trigger ONE upstream fetch, not 20.
   Callers arriving while a fetch is already in flight for the same key
   await that same in-flight call instead of starting their own.

This is a single-process in-memory cache (module-level dict), matching the
pattern already used for bias calibration (database.py's _bias_cache) and
the Safe Bets sweep (predictor.py's _safe_bets_cache) — appropriate for
this app's single-instance Railway deployment. It does NOT persist across
restarts/deploys, which is fine: worst case after a deploy is one full
price of cold-cache fetches, same as today.
"""

import asyncio
import functools
import time
from datetime import date as _date

_store: dict = {}      # key -> (expires_at_monotonic, value)
_inflight: dict = {}   # key -> asyncio.Task, for request coalescing


def is_today(target_date) -> bool:
    """True if target_date (an ISO string or None) is today. None is treated
    as today since that's what every ESPN fetch function defaults to."""
    if not target_date:
        return True
    return target_date == _date.today().isoformat()


async def cached(key, ttl: int, fetch):
    """
    Return the cached value for `key` if still fresh, otherwise call the
    zero-arg async `fetch()` callable, cache its result for `ttl` seconds,
    and return it. Concurrent callers for the same key while a fetch is
    already running share that one in-flight call instead of duplicating it.
    """
    now = time.monotonic()
    entry = _store.get(key)
    if entry is not None and entry[0] > now:
        return entry[1]

    existing = _inflight.get(key)
    if existing is not None:
        return await existing

    task = asyncio.ensure_future(fetch())
    _inflight[key] = task
    try:
        result = await task
        _store[key] = (time.monotonic() + ttl, result)
        return result
    finally:
        _inflight.pop(key, None)


def espn_cache(ttl_fn):
    """
    Decorator for an async ESPN-fetching function: caches its result keyed
    by its own name + arguments, with a TTL computed per-call by
    ttl_fn(*args, **kwargs) — so, e.g., "today's fixtures" can get a short
    TTL while "last month's fixtures" gets a long one, from the same
    decorated function.

    A failed fetch (exception) is never cached — only successful results
    are stored, so a transient ESPN error (see the rate-limit note above)
    doesn't get "frozen" as the cached answer for the next several minutes.
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = (fn.__module__, fn.__qualname__, args, tuple(sorted(kwargs.items())))
            ttl = ttl_fn(*args, **kwargs)
            return await cached(key, ttl, lambda: fn(*args, **kwargs))
        return wrapper
    return decorator
