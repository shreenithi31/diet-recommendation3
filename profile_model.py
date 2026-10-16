"""ML model that picks the best macro strategy for a person.

Trained on synthetic profiles labelled with a guideline-based policy (+ label noise),
so it learns a smooth decision boundary. Swap in dietitian-labelled data for production.
"""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from nutrition import ACTIVITY_FACTORS, GOALS, Profile, bmi, rule_strategy, STRATEGIES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "profile_model.joblib"
ACT = list(ACTIVITY_FACTORS)
FEATURES = ["age", "bmi", "male", "activity_idx", "goal_idx", "diabetes", "hypertension", "cholesterol"]


def synth(n=8000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(14, 80, n),
        "bmi": rng.normal(25, 5, n).clip(15, 45),
        "male": rng.integers(0, 2, n),
        "activity_idx": rng.integers(0, 5, n),
        "goal_idx": rng.integers(0, 3, n),
        "diabetes": rng.random(n) < 0.12,
        "hypertension": rng.random(n) < 0.18,
        "cholesterol": rng.random(n) < 0.15,
    })
    for c in ("diabetes", "hypertension", "cholesterol"):
        df[c] = df[c].astype(int)
    y = df.apply(lambda r: rule_strategy(r.age, r.bmi, r.activity_idx, r.goal_idx,
                                         r.diabetes, r.hypertension, r.cholesterol), axis=1)
    noise = rng.random(n) < 0.04                      # 4% label noise
    y[noise] = rng.choice(list(STRATEGIES), noise.sum())
    return df, y


def train():
    X, y = synth()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=1).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"Strategy classifier accuracy: {accuracy_score(yte, pred):.3f}")
    print(classification_report(yte, pred, zero_division=0))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def _row(p: Profile):
    return pd.DataFrame([{
        "age": p.age, "bmi": bmi(p.weight_kg, p.height_cm), "male": int(p.sex == "male"),
        "activity_idx": ACT.index(p.activity), "goal_idx": GOALS.index(p.goal),
        "diabetes": int("diabetes" in p.conditions),
        "hypertension": int("hypertension" in p.conditions),
        "cholesterol": int("high_cholesterol" in p.conditions)}])[FEATURES]


@lru_cache(maxsize=1)
def _load():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict_strategy(p: Profile) -> str:
    p.validate()
    model = _load()
    row = _row(p)
    if model is None:   # fallback to rules if the model hasn't been trained
        r = row.iloc[0]
        return rule_strategy(r.age, r.bmi, r.activity_idx, r.goal_idx,
                             r.diabetes, r.hypertension, r.cholesterol)
    return str(model.predict(row)[0])


if __name__ == "__main__":
    train()
