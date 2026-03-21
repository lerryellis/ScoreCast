import os
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY    = os.getenv("API_FOOTBALL_KEY", "")
FOOTBALL_DATA_KEY   = os.getenv("FOOTBALL_DATA_KEY", "c620ffef901d44df957dc6aa21d519f6")
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
    "Ghana Premier League": "gha.1",
}

# NBA is the only basketball league for now
BASKETBALL_LEAGUE = "NBA"

# Model settings
FOOTBALL_FORM_WINDOW  = 10   # last N matches for rolling averages
BASKETBALL_FORM_WINDOW = 10
HOME_ADVANTAGE_FACTOR  = 1.15  # ~15% boost for home team in football

# Cache TTL in seconds
CACHE_TTL_FIXTURES   = 1800   # 30 min
CACHE_TTL_TEAM_STATS = 86400  # 24 hours
