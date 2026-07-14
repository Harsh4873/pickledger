from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipl.data.squad_fetcher import fetch_current_squads, get_available_players
from ipl.data_loader import _default_db_path


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "win_predictor.pkl"
FEATURES_PATH = MODEL_DIR / "win_features.json"
CUTOFF_DATE = pd.Timestamp("2024-01-01")
FEATURE_COLS = [
    "h2h_team1_rate",
    "team1_form_rate",
    "team2_form_rate",
    "venue_team1_rate",
    "venue_team2_rate",
    "toss_team1",
    "toss_bat",
]
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


try:
    import joblib
except ImportError:
    import pickle

    class _JoblibCompat:
        @staticmethod
        def dump(obj: Any, path: str | Path) -> None:
            with Path(path).open("wb") as handle:
                pickle.dump(obj, handle)

        @staticmethod
        def load(path: str | Path) -> Any:
            with Path(path).open("rb") as handle:
                return pickle.load(handle)

    joblib = _JoblibCompat()


class LogisticThresholdClassifier:
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 4000,
        reg_strength: float = 0.001,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_strength = reg_strength
        self.backend_name = "logistic_threshold"

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))

    def fit(self, X: Any, y: Any) -> "LogisticThresholdClassifier":
        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.feature_mean_ = x_arr.mean(axis=0)
        self.feature_std_ = x_arr.std(axis=0)
        self.feature_std_[self.feature_std_ == 0] = 1.0
        x_scaled = (x_arr - self.feature_mean_) / self.feature_std_

        weights = np.zeros(x_scaled.shape[1], dtype=float)
        bias = 0.0
        for _ in range(self.max_iter):
            probs = self._sigmoid(x_scaled @ weights + bias)
            grad_w = (x_scaled.T @ (probs - y_arr)) / len(y_arr)
            grad_w = grad_w + self.reg_strength * weights
            grad_b = float(np.mean(probs - y_arr))
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

        self.weights_ = weights
        self.bias_ = bias

        train_probs = self.predict_proba(x_arr)[:, 1]
        thresholds = np.linspace(0.35, 0.65, 61)
        accuracies = [float(np.mean((train_probs >= t).astype(int) == y_arr)) for t in thresholds]
        self.threshold_ = float(thresholds[int(np.argmax(accuracies))])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float)
        x_scaled = (x_arr - self.feature_mean_) / self.feature_std_
        probs = self._sigmoid(x_scaled @ self.weights_ + self.bias_)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def _canonical_team(name: Any) -> str | None:
    if name is None:
        return None
    text = " ".join(str(name).split()).strip()
    if not text:
        return None
    return TEAM_ALIASES.get(text, text)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _load_matches(db_path: str | Path) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        matches = pd.read_sql_query(
            """
            SELECT
                match_id,
                season,
                date,
                venue,
                team1,
                team2,
                toss_winner,
                toss_decision,
                winner
            FROM ipl_matches
            WHERE team1 IS NOT NULL
              AND team2 IS NOT NULL
            """,
            con,
        )
    finally:
        con.close()

    matches = matches.copy()
    for col in ("team1", "team2", "toss_winner", "winner"):
        matches[col] = matches[col].map(_canonical_team)
    matches["venue"] = matches["venue"].map(_normalize_text)
    matches["toss_decision"] = matches["toss_decision"].map(_normalize_text)
    matches["match_id"] = matches["match_id"].astype(str)
    matches["date"] = pd.to_datetime(
        matches["date"].astype(str).str.replace("/", "-", regex=False),
        errors="coerce",
    )
    matches = matches.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    return matches


def _build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    team1_rows = matches.assign(
        team=matches["team1"],
        opponent=matches["team2"],
        slot="team1",
        won=(matches["winner"] == matches["team1"]).astype("int8"),
    )
    team2_rows = matches.assign(
        team=matches["team2"],
        opponent=matches["team1"],
        slot="team2",
        won=(matches["winner"] == matches["team2"]).astype("int8"),
    )

    appearances = pd.concat([team1_rows, team2_rows], ignore_index=True)
    appearances = appearances.sort_values(["team", "date", "match_id", "slot"]).reset_index(drop=True)

    team_group = appearances.groupby("team", sort=False)["won"]
    appearances["form_wins_last5"] = (
        team_group.transform(lambda values: values.shift(1).rolling(5, min_periods=1).sum()).fillna(0.0)
    )
    appearances["form_rate"] = appearances["form_wins_last5"] / 5.0

    venue_group = appearances.groupby(["team", "venue"], sort=False)
    appearances["venue_matches_before"] = venue_group.cumcount()
    appearances["venue_wins_before"] = venue_group["won"].cumsum() - appearances["won"]
    appearances["venue_rate"] = np.where(
        appearances["venue_matches_before"] > 0,
        appearances["venue_wins_before"] / appearances["venue_matches_before"],
        0.5,
    )

    appearances["pair_a"] = np.where(
        appearances["team"] <= appearances["opponent"],
        appearances["team"],
        appearances["opponent"],
    )
    appearances["pair_b"] = np.where(
        appearances["team"] <= appearances["opponent"],
        appearances["opponent"],
        appearances["team"],
    )
    h2h_group = appearances.groupby(["pair_a", "pair_b", "team"], sort=False)
    appearances["h2h_matches_before"] = h2h_group.cumcount()
    appearances["h2h_wins_before"] = h2h_group["won"].cumsum() - appearances["won"]
    appearances["h2h_rate"] = np.where(
        appearances["h2h_matches_before"] > 0,
        appearances["h2h_wins_before"] / appearances["h2h_matches_before"],
        0.5,
    )

    return appearances


def build_match_features(db_path: str | Path) -> pd.DataFrame:
    matches = _load_matches(db_path)
    matches = matches.dropna(subset=["winner"]).copy()
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "date",
                "team1",
                "team2",
                "season",
                "venue",
                *FEATURE_COLS,
                "target",
            ]
        )

    appearances = _build_team_history(matches)
    team1_features = appearances[appearances["slot"] == "team1"][
        ["match_id", "h2h_rate", "form_rate", "venue_rate"]
    ].rename(
        columns={
            "h2h_rate": "h2h_team1_rate",
            "form_rate": "team1_form_rate",
            "venue_rate": "venue_team1_rate",
        }
    )
    team2_features = appearances[appearances["slot"] == "team2"][
        ["match_id", "form_rate", "venue_rate"]
    ].rename(
        columns={
            "form_rate": "team2_form_rate",
            "venue_rate": "venue_team2_rate",
        }
    )

    features = (
        matches.merge(team1_features, on="match_id", how="left")
        .merge(team2_features, on="match_id", how="left")
        .copy()
    )
    features["h2h_team1_rate"] = features["h2h_team1_rate"].fillna(0.5)
    features["team1_form_rate"] = features["team1_form_rate"].fillna(0.0)
    features["team2_form_rate"] = features["team2_form_rate"].fillna(0.0)
    features["venue_team1_rate"] = features["venue_team1_rate"].fillna(0.5)
    features["venue_team2_rate"] = features["venue_team2_rate"].fillna(0.5)
    features["toss_team1"] = (features["toss_winner"] == features["team1"]).astype(int)
    features["toss_bat"] = features["toss_decision"].fillna("").str.lower().eq("bat").astype(int)
    features["target"] = (features["winner"] == features["team1"]).astype(int)

    return features[
        [
            "match_id",
            "date",
            "team1",
            "team2",
            "season",
            "venue",
            *FEATURE_COLS,
            "target",
        ]
    ]


def _build_model() -> Any:
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.backend_name = "xgboost"
        return model
    except ImportError:
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            try:
                model = GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
            except TypeError:
                model = GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
            model.backend_name = "gradient_boosting"
            return model
        except ImportError:
            return LogisticThresholdClassifier()


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    truth = np.asarray(y_true, dtype=float)
    return float(-(truth * np.log(probs) + (1 - truth) * np.log(1 - probs)).mean())


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    probs = np.asarray(y_prob, dtype=float)
    return float(np.mean((probs - truth) ** 2))


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    lines = ["              precision    recall  f1-score   support"]
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)

    for label in (0, 1):
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        support = int(np.sum(truth == label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        lines.append(
            f"{label:>12} {precision:>10.2f} {recall:>8.2f} {f1:>9.2f} {support:>9}"
        )

    accuracy = float(np.mean(truth == pred))
    lines.append("")
    lines.append(f"    accuracy {accuracy:>29.2f} {len(truth):>9}")
    return "\n".join(lines)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.3, 0.7, 81)
    accuracies = [float(np.mean((probs >= threshold).astype(int) == truth)) for threshold in thresholds]
    return float(thresholds[int(np.argmax(accuracies))])


def train_win_model(db_path: str | Path):
    feature_frame = build_match_features(db_path)
    train_df = feature_frame[feature_frame["date"] < CUTOFF_DATE].copy()
    test_df = feature_frame[feature_frame["date"] >= CUTOFF_DATE].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set")

    x_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["target"].to_numpy(dtype=int)
    x_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df["target"].to_numpy(dtype=int)

    model = _build_model()
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    y_prob = model.predict_proba(x_test)[:, 1]
    threshold = getattr(model, "threshold_", _best_threshold(y_train, train_prob))
    setattr(model, "threshold_", float(threshold))
    y_pred = (y_prob >= float(threshold)).astype(int)
    accuracy = float(np.mean(y_pred == y_test))
    loss = _log_loss(y_test, y_prob)
    brier = _brier_score(y_test, y_prob)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with FEATURES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(FEATURE_COLS, handle, indent=2)

    print(f"Model backend: {getattr(model, 'backend_name', type(model).__name__)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Decision threshold: {float(threshold):.3f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log loss: {loss:.4f}")
    print(f"Brier score: {brier:.4f}")
    print("Classification report:")
    print(_classification_report(y_test, y_pred))

    return model, accuracy, loss


def _compute_inference_features(
    matches: pd.DataFrame,
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
) -> pd.DataFrame:
    history = matches.dropna(subset=["winner"]).copy()
    venue = _normalize_text(venue) or ""
    team1 = _canonical_team(team1) or team1
    team2 = _canonical_team(team2) or team2
    toss_winner = _canonical_team(toss_winner) or toss_winner
    toss_decision = (_normalize_text(toss_decision) or "").lower()

    h2h_mask = (
        ((history["team1"] == team1) & (history["team2"] == team2))
        | ((history["team1"] == team2) & (history["team2"] == team1))
    )
    h2h_matches = history[h2h_mask]
    h2h_team1_rate = 0.5 if h2h_matches.empty else float((h2h_matches["winner"] == team1).mean())

    def _team_form_rate(team: str) -> float:
        team_matches = history[(history["team1"] == team) | (history["team2"] == team)].sort_values(
            ["date", "match_id"]
        )
        recent = team_matches.tail(5)
        if recent.empty:
            return 0.0
        return float((recent["winner"] == team).sum() / 5.0)

    def _venue_rate(team: str) -> float:
        venue_matches = history[
            (history["venue"] == venue) & ((history["team1"] == team) | (history["team2"] == team))
        ]
        if venue_matches.empty:
            return 0.5
        return float((venue_matches["winner"] == team).mean())

    return pd.DataFrame(
        [
            {
                "h2h_team1_rate": h2h_team1_rate,
                "team1_form_rate": _team_form_rate(team1),
                "team2_form_rate": _team_form_rate(team2),
                "venue_team1_rate": _venue_rate(team1),
                "venue_team2_rate": _venue_rate(team2),
                "toss_team1": int(toss_winner == team1),
                "toss_bat": int(toss_decision == "bat"),
            }
        ],
        columns=FEATURE_COLS,
    )


def predict_winner(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    db_path: str | Path,
) -> dict[str, Any]:
    build_match_features(db_path)
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        train_win_model(db_path)

    model = joblib.load(MODEL_PATH)
    with FEATURES_PATH.open("r", encoding="utf-8") as handle:
        feature_cols = json.load(handle)

    matches = _load_matches(db_path)
    feature_row = _compute_inference_features(matches, team1, team2, venue, toss_winner, toss_decision)
    probabilities = model.predict_proba(feature_row[feature_cols].to_numpy(dtype=float))[0]
    team1_prob = float(probabilities[1])
    team2_prob = float(probabilities[0])
    predicted_winner = team1 if team1_prob >= team2_prob else team2
    top_prob = max(team1_prob, team2_prob)
    confidence = "HIGH" if top_prob > 0.65 else "MEDIUM" if top_prob > 0.55 else "LOW"

    return {
        "team1": _canonical_team(team1) or team1,
        "team2": _canonical_team(team2) or team2,
        "team1_win_prob": team1_prob,
        "team2_win_prob": team2_prob,
        "predicted_winner": predicted_winner,
        "confidence": confidence,
    }


if __name__ == "__main__":
    db_path = _default_db_path()
    fetch_current_squads()
    _, accuracy, loss = train_win_model(db_path)
    print(f"\nSummary: accuracy={accuracy:.4f}, logloss={loss:.4f}")

    samples = [
        (
            "Mumbai Indians",
            "Chennai Super Kings",
            "Wankhede Stadium",
            "Mumbai Indians",
            "bat",
        ),
        (
            "Royal Challengers Bengaluru",
            "Kolkata Knight Riders",
            "M Chinnaswamy Stadium",
            "Kolkata Knight Riders",
            "field",
        ),
        (
            "Rajasthan Royals",
            "Sunrisers Hyderabad",
            "Sawai Mansingh Stadium",
            "Rajasthan Royals",
            "bat",
        ),
    ]

    for team1, team2, venue, toss_winner, toss_decision in samples:
        prediction = predict_winner(team1, team2, venue, toss_winner, toss_decision, db_path)
        print(f"\nPrediction: {prediction['team1']} vs {prediction['team2']} at {venue}")
        print(
            f"  team1_win_prob={prediction['team1_win_prob']:.4f}, "
            f"team2_win_prob={prediction['team2_win_prob']:.4f}, "
            f"predicted_winner={prediction['predicted_winner']}, "
            f"confidence={prediction['confidence']}"
        )
        for team in (prediction["team1"], prediction["team2"]):
            players = get_available_players(team, db_path)
            print(f"  {team} available players ({len(players)}):")
            for player in players:
                print(f"    {player}")
