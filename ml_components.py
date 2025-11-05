from typing import List, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# --------- Card helpers ---------
VALUES: Dict[str, int] = {"A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10}
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]


def parse_cards(s: str) -> List[str]:
    parts = [c.strip().upper() for c in str(s).split(",") if str(s) != "nan"]
    return [c for c in parts if c]


def hand_value(cards_csv: str) -> int:
    cards = parse_cards(cards_csv)
    vals = []
    for c in cards:
        vals.append(VALUES.get(c, 0) if c != "A" else 11)
    total = sum(vals)
    aces = cards.count("A")
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def dealer_up_rank(cards_csv: str) -> str:
    cards = parse_cards(cards_csv)
    return cards[0] if cards else ""


def _build_cards_before_action(group: pd.DataFrame) -> pd.Series:
    final_cards = parse_cards(group["player_cards"].iloc[0])
    initial_two = final_cards[:2]
    extras = final_cards[2:]

    is_hit = (group["action"].str.lower() == "hit").astype(int)
    is_double = (group["action"].str.lower() == "double").astype(int)
    draws_prev = (is_hit + is_double).cumsum().shift(1, fill_value=0).astype(int)

    before_list = []
    for k in draws_prev.to_list():
        k = max(0, int(k))
        extra_taken = extras[: min(k, len(extras))]
        before_list.append(",".join(initial_two + extra_taken))
    return pd.Series(before_list, index=group.index, dtype="string")


def expand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "game_id" not in d.columns:
        d.insert(0, "game_id", range(1, len(d) + 1))
    d["actions"] = d["actions"].fillna("").astype(str)
    d["action"] = d["actions"].str.split(",")
    d = d.explode("action").copy()
    d["action"] = d["action"].fillna("").str.strip().str.lower()
    d = d.sort_values(["game_id"]).copy()
    d["step"] = d.groupby("game_id").cumcount() + 1
    d = d.sort_values(["game_id", "step"]).copy()
    d["player_cards_step"] = d.groupby("game_id", group_keys=False).apply(_build_cards_before_action)
    d["dealer_visible_card"] = d["dealer_cards"].apply(dealer_up_rank)
    return d


# --------- Transformers and pipeline ---------
class DropColumns(BaseEstimator, TransformerMixin):
    def _init_(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = [c for c in self.columns_to_drop if c in X.columns]
        return X.drop(columns=cols)


class BlackjackFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        src_col = "player_cards_step" if "player_cards_step" in X_.columns else "player_cards"
        X_["player_total"] = X_[src_col].apply(hand_value)
        X_["player_aces"] = X_[src_col].apply(lambda s: parse_cards(str(s)).count("A"))

        def dealer_val(cards):
            r = dealer_up_rank(cards)
            if r == "A":
                return 11
            return VALUES.get(r, 0)

        X_["dealer_visible"] = X_["dealer_cards"].apply(dealer_val)
        return X_


NUM_PIPE = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])

COLUMN_TRANSFORM = ColumnTransformer([
    ("num", NUM_PIPE, ["player_total", "player_aces", "dealer_visible", "step"]),
], remainder="drop")


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("feat", BlackjackFeatureExtractor()),
        ("coltrans", COLUMN_TRANSFORM),
    ])


def build_model() -> Pipeline:
    model = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
    pipe = Pipeline([
        ("pre", make_preprocessor()),
        ("model", model),
        ])
    return pipe