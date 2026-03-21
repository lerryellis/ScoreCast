import os
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
ADMIN_KEY        = os.getenv("ADMIN_KEY", "betscore-admin")
PORT             = int(os.getenv("PORT", 8000))

# Supported leagues for football predictions
FOOTBALL_LEAGUES = {
    "Premier League":     {"id": 39,  "country": "England"},
    "La Liga":            {"id": 140, "country": "Spain"},
    "Serie A":            {"id": 135, "country": "Italy"},
    "Bundesliga":         {"id": 78,  "country": "Germany"},
    "Ligue 1":            {"id": 61,  "country": "France"},
    "Champions League":   {"id": 2,   "country": "Europe"},
    "Ghana Premier League": {"id": 169, "country": "Ghana"},
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
