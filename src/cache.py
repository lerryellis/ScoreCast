"""
TTL cache for ESPN fetches — in-memory fast path backed by a persistent
Supabase layer, with request coalescing.

Why this exists: every page load / refresh-button click re-fetches fixtures,
standings, team form, H2H, and calendar data from ESPN — with dozens of
concurrent users (or just one user refreshing repeatedly), that's a lot of
avoidable traffic to a free, unauthenticated, rate-limited public API (see
CLAUDE.md's "Known issue: ESPN occasionally 403s" note — this is the actual
fix for that, not a header trick).

Three things this needs to get right:
1. Adaptive TTL — a live match's score needs to look fresh within a minute
   or so; a team's "last 5 games" barely changes within a whole day. One
   flat TTL for everything is either too slow for live data or needlessly
   re-fetches static data. Each cached function gets its own TTL policy
   (see espn_cache's ttl_fn param) instead of a single global number.
2. Request coalescing — if 20 users refresh at the same moment right after
   a cache entry expires, they should trigger ONE upstream fetch, not 20.
   Callers arriving while a fetch is already in flight for the same key
   await that same in-flight call instead of starting their own.
3. Persistence across restarts — the in-memory layer (module-level dict) is
   wiped on every Railway restart/redeploy, which happens often during
   active development. Supabase's `espn_cache` table backs it: on an
   in-memory miss, check there before ever calling ESPN, so a freshly
   booted process can still serve anything still within its TTL from
   before the restart instead of everyone hitting a cold cache. The
   in-memory layer stays the fast path (no network at all on a hit) —
   Supabase is only consulted on an in-memory miss.
"""

import asyncio
import functools
import json
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


async def cached(key: str, ttl: int, fetch):
    """
    Return the cached value for `key` if still fresh (checking the
    in-memory layer, then Supabase), otherwise call the zero-arg async
    `fetch()` callable, persist its result, and return it. Concurrent
    callers for the same key while a fetch is already running share that
    one in-flight call instead of duplicating it.
    """
    now = time.monotonic()
    entry = _store.get(key)
    if entry is not None and entry[0] > now:
        return entry[1]

    existing = _inflight.get(key)
    if existing is not None:
        return await existing

    async def _resolve():
        # Lazy import: src.database imports src.fetcher (which imports this
        # module) at call time in various places, so importing it at
        # cache.py's module level would create a circular import.
        from src.database import get_espn_cache_entry, set_espn_cache_entry

        db_hit = await get_espn_cache_entry(key)
        if db_hit is not None:
            value, remaining = db_hit
            _store[key] = (time.monotonic() + remaining, value)
            return value

        result = await fetch()
        _store[key] = (time.monotonic() + ttl, result)
        await set_espn_cache_entry(key, result, ttl)
        return result

    task = asyncio.ensure_future(_resolve())
    _inflight[key] = task
    try:
        return await task
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
            # A JSON string (not a raw tuple) so it doubles as the Supabase
            # primary key — args/kwargs are always plain strings/ints/None
            # across every decorated function, so this is always valid JSON.
            key = json.dumps(
                [fn.__module__, fn.__qualname__, list(args), sorted(kwargs.items())],
                sort_keys=True, default=str,
            )
            ttl = ttl_fn(*args, **kwargs)
            return await cached(key, ttl, lambda: fn(*args, **kwargs))
        return wrapper
    return decorator
