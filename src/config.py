import os
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY    = os.getenv("API_FOOTBALL_KEY", "")
FOOTBALL_DATA_KEY   = os.getenv("FOOTBALL_DATA_KEY", "c620ffef901d44df957dc6aa21d519f6")
SUPABASE_URL        = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY        = os.getenv("SUPABASE_KEY", "")
THESPORTSDB_KEY     = os.getenv("THESPORTSDB_KEY", "123")   # free public key
ADMIN_KEY           = os.getenv("ADMIN_KEY", "betscore-admin")
PORT                = int(os.getenv("PORT", 8000))

# football-data.org competition codes for leagues we support
FOOTBALL_DATA_COMPETITIONS = {
    "Premier League":       "PL",
    "Championship":         "ELC",
    "La Liga":              "PD",
    "Serie A":              "SA",
    "Bundesliga":           "BL1",
    "Ligue 1":              "FL1",
    "Champions League": "CL",
    # Europa League not on free tier; Ghana PL not covered
}

# Supported leagues for football predictions (API-Football IDs — kept for reference)
FOOTBALL_LEAGUES = {
    "Premier League":       {"id": 39,  "country": "England"},
    "La Liga":              {"id": 140, "country": "Spain"},
    "Serie A":              {"id": 135, "country": "Italy"},
    "Bundesliga":           {"id": 78,  "country": "Germany"},
    "Ligue 1":              {"id": 61,  "country": "France"},
    "Champions League": {"id": 2, "country": "Europe"},
    "Ghana Premier League": {"id": 169, "country": "Ghana"},
}

# ESPN league slugs — used as primary data source (free, no key, current season)
ESPN_FOOTBALL_LEAGUES = {
    "Premier League":       "eng.1",
    "Championship":         "eng.2",
    "La Liga":              "esp.1",
    "Serie A":              "ita.1",
    "Bundesliga":           "ger.1",
    "Ligue 1":              "fra.1",
    "Champions League":     "UEFA.CHAMPIONS",
    "Europa League":        "UEFA.EUROPA",
    "FA Cup":               "eng.fa",
    "EFL Cup":              "eng.league_cup",
    "Copa del Rey":         "esp.copa_del_rey",
    "Coppa Italia":         "ita.coppa_italia",
    "DFB Pokal":            "ger.dfb_pokal",
    "Coupe de France":      "fra.coupe_de_france",
    "Ghana Premier League": "gha.1",
}

# Some competitions run their pre-tournament stage under a SEPARATE ESPN
# slug from the main one — the Champions League's qualifying/play-off
# rounds (knockout ties played before the league phase even exists) use
# distinct slugs from the league-phase competition itself. Verified live
# (2026-08-26): ESPN_FOOTBALL_LEAGUES' plain "UEFA.CHAMPIONS"/
# "uefa.wchampions" slugs had zero fixtures on a day when real play-off
# second legs were being played (Fenerbahçe vs Lyon, Real Madrid vs Ajax,
# etc.) — those were under "UEFA.CHAMPIONS_QUAL"/"uefa.wchampions_qual"
# instead. Leagues listed here get fixtures merged from every slug in the
# list (see get_all_football_predictions) rather than just the primary one,
# so the competition shows real games year-round instead of going empty
# for the ~6 weeks/year the tournament is in its qualifying rounds.
#
# "Champions League" is one dropdown entry covering all four slugs (men's +
# women's, league phase + qualifying) rather than separate Men/Women
# entries — keeps the league dropdown from growing indefinitely. Individual
# matches are tagged is_women (see WOMENS_SLUGS) so the UI can still tell
# them apart on the match card itself.
MULTI_SLUG_LEAGUES = {
    "Champions League": [
        "UEFA.CHAMPIONS", "UEFA.CHAMPIONS_QUAL",
        "uefa.wchampions", "uefa.wchampions_qual",
    ],
    # Same qualifying-slug split confirmed for Europa League (verified live:
    # play-off second legs on 2026-08-27 were 0 under UEFA.EUROPA alone, 12
    # fixtures under UEFA.EUROPA_QUAL). No Women's Europa League exists.
    "Europa League": ["UEFA.EUROPA", "UEFA.EUROPA_QUAL"],
}

# Slugs that belong to a women's competition — used to tag is_women on each
# fixture (see predictor.get_all_football_predictions) so match cards can
# show a "W" badge without needing a separate dropdown entry per gender.
WOMENS_SLUGS = {"uefa.wchampions", "uefa.wchampions_qual"}

# ESPN's scoreboard endpoint returns its own display name for each league
# (e.g. "English Premier League", "Spanish LALIGA") which is what gets stored
# on every fixture/prediction row — NOT the short key above. Bias calibration
# groups by that stored name, so anything doing a per-league bias lookup by
# the short key (e.g. "Premier League") must normalize through this map first,
# or it silently misses and falls back to the global average. Verified against
# a live ESPN scoreboard call per league (2026-08-21).
ESPN_LEAGUE_DISPLAY_NAME = {
    "Premier League":       "English Premier League",
    "Championship":         "English League Championship",
    "La Liga":              "Spanish LALIGA",
    "Serie A":              "Italian Serie A",
    "Bundesliga":           "German Bundesliga",
    "Ligue 1":              "French Ligue 1",
    "Ghana Premier League": "Ghanaian Premier League",
    # No "Champions League" entry: that dropdown key now covers 4 different
    # ESPN display names (men's/women's x league-phase/qualifying), so a
    # 1:1 mapping doesn't apply — normalize_league_name() falls through
    # unchanged for each, and bias calibration buckets them separately
    # (arguably more correct anyway: qualifying-round and league-phase
    # scoring patterns differ enough to not want them pooled).
}

# Reverse of the above — maps ESPN's stored display name back to our short
# canonical key. Used to normalize league names when grouping calibration
# data, so historical rows (stored under ESPN's name) match the short key
# that predictor.py looks bias up by.
ESPN_DISPLAY_TO_LEAGUE = {v: k for k, v in ESPN_LEAGUE_DISPLAY_NAME.items()}


def normalize_league_name(name: str) -> str:
    """Map an ESPN display name (as stored on fixtures/predictions) back to
    our canonical short league key, if known. Falls through unchanged for
    names we don't have a mapping for (e.g. cup competitions, "Unknown")."""
    return ESPN_DISPLAY_TO_LEAGUE.get(name, name)


# Feeder league one tier down, for every division we track — used to pull
# "recent form" for a newly-promoted team with no match history yet at its
# current level. Chains multiple hops (e.g. eng.1 -> eng.2 -> eng.3), so a
# team promoted two divisions in one close (League One -> Championship, then
# Championship -> Premier League the year after) still resolves correctly —
# _schedule_with_season_fallback walks this chain until it finds enough
# matches or runs out of tiers. Verified live against ESPN scoreboard
# (2026-08): eng.2 "English League Championship", eng.3 "English League One",
# eng.4 "English League Two", esp.2 "Spanish LALIGA 2", ita.2 "Italian Serie B",
# ger.2 "German 2. Bundesliga", fra.2 "French Ligue 2".
PROMOTION_FEEDER_LEAGUE = {
    "eng.1": "eng.2",
    "eng.2": "eng.3",
    "eng.3": "eng.4",
    "esp.1": "esp.2",
    "ita.1": "ita.2",
    "ger.1": "ger.2",
    "fra.1": "fra.2",
}

# Map cup slugs to their parent league — for form data, use league not cup results
CUP_TO_LEAGUE = {
    "eng.fa":              "eng.1",
    "eng.league_cup":      "eng.1",
    "esp.copa_del_rey":    "esp.1",
    "ita.coppa_italia":    "ita.1",
    "ger.dfb_pokal":       "ger.1",
    "fra.coupe_de_france": "fra.1",
}

# Cup competitions per country — used to fetch full schedule for rest/congestion
ESPN_CUP_SLUGS = {
    "eng.1":  ["eng.fa", "eng.league_cup", "UEFA.CHAMPIONS", "UEFA.EUROPA"],
    "eng.2":  ["eng.fa", "eng.league_cup"],
    "esp.1":  ["esp.copa_del_rey", "UEFA.CHAMPIONS", "UEFA.EUROPA"],
    "ita.1":  ["ita.coppa_italia", "UEFA.CHAMPIONS", "UEFA.EUROPA"],
    "ger.1":  ["ger.dfb_pokal", "UEFA.CHAMPIONS", "UEFA.EUROPA"],
    "fra.1":  ["fra.coupe_de_france", "UEFA.CHAMPIONS", "UEFA.EUROPA"],
    "UEFA.CHAMPIONS":   [],  # already a cup competition
    "UEFA.EUROPA":      [],
    "uefa.wchampions":  [],  # UEFA Women's Champions League — no domestic-cup crossover tracked
    "gha.1":  [],
}

# ESPN international football slugs — standalone from club football
ESPN_INTERNATIONAL_LEAGUES = {
    "World Cup 2026":           "fifa.world",
    "UEFA WC Qualifiers":       "fifa.worldq.uefa",
    "CONMEBOL WC Qualifiers":   "fifa.worldq.conmebol",
    "CAF WC Qualifiers":        "fifa.worldq.caf",
    "UEFA Nations League":      "uefa.nations",
    "International Friendlies": "fifa.friendly",
}

# Friendly-competition slugs to exclude when building "current form" and H2H
# for national teams — friendlies are squad-rotation/experimental affairs, so
# they're treated as preseason and dropped rather than counted as real
# results. Kept in ESPN_INTERNATIONAL_LEAGUES/dropdown since they're still a
# real, schedulable competition — just not a form/H2H signal.
FRIENDLY_COMP_SLUGS = {"fifa.friendly"}

# Broader set used only for building a national team's match history + H2H —
# includes the major continental tournaments and every confederation's WC
# qualifiers, which are the richest sources of head-to-head meetings. These are
# NOT shown in the prediction dropdown (that stays ESPN_INTERNATIONAL_LEAGUES).
INTERNATIONAL_COMP_SLUGS = list(dict.fromkeys(
    list(ESPN_INTERNATIONAL_LEAGUES.values()) + [
        "fifa.worldq.concacaf",   # CONCACAF WC qualifiers
        "fifa.worldq.afc",        # AFC (Asia) WC qualifiers
        "fifa.worldq.ofc",        # OFC (Oceania) WC qualifiers
        "uefa.euro",              # UEFA European Championship
        "conmebol.america",       # Copa América
        "caf.nations",            # Africa Cup of Nations
        "concacaf.gold",          # CONCACAF Gold Cup
        "afc.asian.cup",          # AFC Asian Cup
        "concacaf.nations.league",# CONCACAF Nations League
    ]
))

# NBA is the only basketball league for now
BASKETBALL_LEAGUE = "NBA"

# Model settings
FOOTBALL_FORM_WINDOW  = 10   # last N matches for rolling averages
BASKETBALL_FORM_WINDOW = 10
HOME_ADVANTAGE_FACTOR  = 1.15  # ~15% boost for home team in football

# Cache TTL in seconds
CACHE_TTL_FIXTURES   = 1800   # 30 min
CACHE_TTL_TEAM_STATS = 86400  # 24 hours
