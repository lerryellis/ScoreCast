"""
XGBoost ML correction layer.

Architecture:
  - Poisson/basketball model runs first → produces lambdas + full probability distribution
  - ML model takes those outputs as features → predicts corrected lambda_home / lambda_away
  - Poisson re-runs with corrected lambdas → improved probabilities + scoreline
  - Basketball model re-runs with corrected predicted scores → improved output

Using objective="count:poisson" so XGBoost directly models the Poisson rate (expected goals),
not the raw integer count. This is statistically appropriate for score prediction.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FOOTBALL_MODEL_PATH   = MODEL_DIR / "football_ml_latest.pkl"
BASKETBALL_MODEL_PATH = MODEL_DIR / "basketball_ml_latest.pkl"

# Features pulled from DB for training — must match what's available at inference time
FOOTBALL_FEATURES = [
    "lambda_home", "lambda_away",
    "win_prob", "draw_prob", "loss_prob",
    "confidence",
    "predicted_home", "predicted_away",
    "over_0_5", "over_1_5", "over_2_5", "over_3_5",
    "league_enc",
]

BASKETBALL_FEATURES = [
    "predicted_home", "predicted_away",
    "win_prob", "loss_prob",
    "confidence",
]

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="count:poisson",   # models Poisson rate, ideal for goal/point counts
    random_state=42,
    verbosity=0,
)


class SportsMLModel:
    """
    Trains and serves XGBoost corrections for either football or basketball.
    Saves/loads via pickle so the singleton survives restarts.
    """

    def __init__(self, sport: str):
        self.sport          = sport
        self.home_model     = None
        self.away_model     = None
        self.league_encoder: dict = {}
        self.trained        = False
        self.trained_at     = ""
        self.n_samples      = 0
        self.eval_metrics: dict = {}

    # ── Feature construction ───────────────────────────────────────────────────

    def _enc(self, league: str) -> int:
        return self.league_encoder.get(league, 0)

    def build_feature_vector(self, row: dict) -> Optional[list]:
        """
        Build a feature vector from a prediction dict.
        `row` may be a joined DB row (with nested `predictions` key)
        or a flat inference dict — both are handled.
        """
        p = row.get("predictions") or row
        try:
            if self.sport == "football":
                return [
                    float(p.get("lambda_home")     or 1.3),
                    float(p.get("lambda_away")     or 1.0),
                    float(p.get("win_prob")        or 40.0),
                    float(p.get("draw_prob")       or 28.0),
                    float(p.get("loss_prob")       or 32.0),
                    float(p.get("confidence")      or 10.0),
                    float(p.get("predicted_home")  or 1.0),
                    float(p.get("predicted_away")  or 1.0),
                    float(p.get("over_0_5")        or 90.0),
                    float(p.get("over_1_5")        or 75.0),
                    float(p.get("over_2_5")        or 50.0),
                    float(p.get("over_3_5")        or 25.0),
                    float(self._enc(p.get("league", ""))),
                ]
            else:  # basketball
                return [
                    float(p.get("predicted_home")  or 113.0),
                    float(p.get("predicted_away")  or 110.0),
                    float(p.get("win_prob")        or 50.0),
                    float(p.get("loss_prob")       or 50.0),
                    float(p.get("confidence")      or 51.0),
                ]
        except (TypeError, ValueError):
            return None

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, rows: list) -> dict:
        """
        Train on resolved prediction rows joined with actual scores.
        Returns evaluation metrics dict.
        """
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
        from datetime import datetime

        # Build league encoder
        leagues: dict = {}
        for r in rows:
            p = r.get("predictions") or r
            lg = p.get("league", "Unknown") or "Unknown"
            if lg not in leagues:
                leagues[lg] = len(leagues) + 1   # 0 reserved for unknown
        self.league_encoder = leagues

        # Feature matrix
        X, y_home, y_away = [], [], []
        for r in rows:
            vec = self.build_feature_vector(r)
            if vec is None:
                continue
            X.append(vec)
            y_home.append(max(float(r["actual_home"]), 0.0))
            y_away.append(max(float(r["actual_away"]), 0.0))

        n = len(X)
        if n < 15:
            return {"error": f"Insufficient data: {n} samples"}

        X      = np.array(X, dtype=np.float32)
        y_home = np.array(y_home, dtype=np.float32)
        y_away = np.array(y_away, dtype=np.float32)

        self.home_model = XGBRegressor(**XGB_PARAMS)
        self.away_model = XGBRegressor(**XGB_PARAMS)
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)

        # Cross-validated MAE (Poisson objective → output is expected value)
        cv_folds = min(5, max(2, n // 20))
        home_mae = float(-cross_val_score(
            XGBRegressor(**XGB_PARAMS), X, y_home,
            cv=cv_folds, scoring="neg_mean_absolute_error"
        ).mean())
        away_mae = float(-cross_val_score(
            XGBRegressor(**XGB_PARAMS), X, y_away,
            cv=cv_folds, scoring="neg_mean_absolute_error"
        ).mean())

        # Outcome accuracy on training set
        ph = self.home_model.predict(X)
        pa = self.away_model.predict(X)
        outcome_hits = sum(
            1 for p_h, p_a, a_h, a_a in zip(ph, pa, y_home, y_away)
            if (round(p_h) > round(p_a)) == (a_h > a_a)
            or (round(p_h) == round(p_a)) == (a_h == a_a)
        )
        outcome_acc = round(outcome_hits / n * 100, 1)

        self.trained     = True
        self.trained_at  = datetime.utcnow().isoformat()
        self.n_samples   = n
        self.eval_metrics = {
            "n_samples":   n,
            "home_mae_cv": round(home_mae, 3),
            "away_mae_cv": round(away_mae, 3),
            "outcome_acc": outcome_acc,
            "trained_at":  self.trained_at,
        }
        return self.eval_metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, row: dict) -> Optional[Tuple[float, float]]:
        """
        Return (corrected_lh, corrected_la) for football,
        or (corrected_home_pts, corrected_away_pts) for basketball.
        Returns None if model not trained or output is outside realistic range.
        """
        if not self.trained:
            return None
        vec = self.build_feature_vector(row)
        if vec is None:
            return None
        X = np.array([vec], dtype=np.float32)
        lh = float(self.home_model.predict(X)[0])
        la = float(self.away_model.predict(X)[0])

        # Sport-specific sanity clamps — Poisson log-link can extrapolate to
        # extreme values when inputs are outside the training distribution.
        # Reject the ML correction entirely if output is unrealistic.
        if self.sport == "football":
            if not (0.05 <= lh <= 8.0) or not (0.05 <= la <= 8.0):
                return None
        else:  # basketball
            if not (70 <= lh <= 160) or not (70 <= la <= 160):
                return None

        return lh, la

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> Optional["SportsMLModel"]:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, cls) else None
        except Exception:
            return None


# ── In-memory singletons ───────────────────────────────────────────────────────

_football_ml:   Optional[SportsMLModel] = None
_basketball_ml: Optional[SportsMLModel] = None


def get_football_ml() -> Optional[SportsMLModel]:
    global _football_ml
    if _football_ml is None and FOOTBALL_MODEL_PATH.exists():
        _football_ml = SportsMLModel.load(FOOTBALL_MODEL_PATH)
    return _football_ml


def get_basketball_ml() -> Optional[SportsMLModel]:
    global _basketball_ml
    if _basketball_ml is None and BASKETBALL_MODEL_PATH.exists():
        _basketball_ml = SportsMLModel.load(BASKETBALL_MODEL_PATH)
    return _basketball_ml


def set_football_ml(model: SportsMLModel):
    global _football_ml
    _football_ml = model


def set_basketball_ml(model: SportsMLModel):
    global _basketball_ml
    _basketball_ml = model
