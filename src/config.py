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
    "Champions League":     "CL",
    # Europa League not on free tier; Ghana PL not covered
}

# Supported leagues for football predictions (API-Football IDs — kept for reference)
FOOTBALL_LEAGUES = {
    "Premier League":       {"id": 39,  "country": "England"},
    "La Liga":              {"id": 140, "country": "Spain"},
    "Serie A":              {"id": 135, "country": "Italy"},
    "Bundesliga":           {"id": 78,  "country": "Germany"},
    "Ligue 1":              {"id": 61,  "country": "France"},
    "Champions League":     {"id": 2,   "country": "Europe"},
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
    "UEFA.CHAMPIONS": [],  # already a cup competition
    "UEFA.EUROPA":    [],
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

# All international slugs — used to build full match history for a national team
INTERNATIONAL_COMP_SLUGS = list(ESPN_INTERNATIONAL_LEAGUES.values())

# NBA is the only basketball league for now
BASKETBALL_LEAGUE = "NBA"

# Model settings
FOOTBALL_FORM_WINDOW  = 10   # last N matches for rolling averages
BASKETBALL_FORM_WINDOW = 10
HOME_ADVANTAGE_FACTOR  = 1.15  # ~15% boost for home team in football

# Cache TTL in seconds
CACHE_TTL_FIXTURES   = 1800   # 30 min
CACHE_TTL_TEAM_STATS = 86400  # 24 hours
