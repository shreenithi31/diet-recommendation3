"""Content-based, preference-aware meal planner with portion optimisation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from nutrition import DIETS, Profile, compute_targets
from profile_model import predict_strategy

FOODS_CSV = Path(__file__).resolve().parents[1] / "data" / "foods.csv"
DIET_LEVEL = {"vegan": 0, "veg": 1, "egg": 2, "nonveg": 3}

# (meal name, share of daily calories, food slots)
MEALS = [("Breakfast", 0.25, ["breakfast", "fruit", "dairy"]),
         ("Lunch",     0.35, ["protein", "carb", "veg"]),
         ("Snack",     0.10, ["snack", "fruit"]),
         ("Dinner",    0.30, ["protein", "carb", "veg"])]
MIN_PORTION, MAX_PORTION = 0.5, 2.5
SLOT_MAX = {"fruit": 2.0, "veg": 2.0, "dairy": 2.0, "snack": 2.0}   # realistic caps


def load_foods(path=FOODS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["allergens"] = df["allergens"].fillna("")
    df["slot_list"] = df["slots"].str.split("|")
    df["allergen_set"] = df["allergens"].apply(lambda s: {a for a in s.split(";") if a})
    df["diet_level"] = df["diet"].map(DIET_LEVEL)
    return df


def _bounded_lstsq(A, b, lo, hi):
    """Fast bounded least squares: solve, clamp violators, re-solve the free variables."""
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    fixed = np.zeros(len(x), bool)
    for _ in range(4):
        bad = ((x < lo) | (x > hi)) & ~fixed
        if not bad.any():
            break
        x[bad] = np.clip(x[bad], lo, hi[bad] if hasattr(hi, "__len__") else hi)
        fixed |= bad
        free = ~fixed
        if not free.any():
            break
        x[free] = np.linalg.lstsq(A[:, free], b - A[:, fixed] @ x[fixed], rcond=None)[0]
    return np.clip(x, lo, hi)


def _softmax(x, tau):
    z = (np.asarray(x, float) - np.max(x)) / tau
    e = np.exp(z)
    return e / e.sum()


class DietRecommender:
    def __init__(self, foods: pd.DataFrame | None = None):
        self.foods = load_foods() if foods is None else foods

    # ---------- filtering ----------
    def eligible(self, p: Profile) -> pd.DataFrame:
        df = self.foods
        m = df["diet_level"] <= DIETS[p.diet]
        allergies = set(p.allergies)
        m &= df["allergen_set"].apply(lambda s: not (s & allergies))
        dislikes = [d.lower().strip() for d in p.dislikes if d.strip()]
        names = df["name"].str.lower()
        if dislikes:
            m &= ~names.apply(lambda n: any(d in n for d in dislikes))
        if "diabetes" in p.conditions:
            m &= df["gi"] != "high"
        if "hypertension" in p.conditions:
            m &= df["sodium_mg"] <= 400
        if "high_cholesterol" in p.conditions:
            m &= df["fat_g"] <= 14
        return df[m].reset_index(drop=True)

    # ---------- scoring ----------
    @staticmethod
    def _preference_bonus(df: pd.DataFrame, likes) -> pd.Series:
        bonus = pd.Series(0.0, index=df.index)
        likes = [l.lower().strip() for l in likes if l.strip()]
        if not likes or df.empty:
            return bonus
        liked = df["name"].str.lower().apply(lambda n: any(l in n for l in likes))
        if not liked.any():
            return bonus
        dens = df[["protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g"]].div(df["calories"] / 100, axis=0)
        X = StandardScaler().fit_transform(dens)
        sim = cosine_similarity(X, X[liked.to_numpy()]).max(axis=1)   # similar to liked foods
        return 0.5 * pd.Series(np.clip(sim, 0, 1), index=df.index) + liked.astype(float)

    @staticmethod
    def _quality(df, p, pref, usage) -> pd.Series:
        fiber_density = (df["fiber_g"] / (df["calories"] / 100)).clip(upper=6)
        sod_w = 1.5 if "hypertension" in p.conditions else 0.5
        sug_w = 2.0 if "diabetes" in p.conditions else 1.0
        q = 1 + 0.12 * fiber_density - sod_w * df["sodium_mg"] / 1000 - sug_w * df["sugar_g"] / 50
        return q + pref - 0.7 * df["name"].map(usage).fillna(0)

    # ---------- portion optimisation ----------
    @staticmethod
    def _fit(M: np.ndarray, target: dict, max_p=None):
        """M: (items x 4) matrix of calories, protein, carbs, fat per serving."""
        w = np.array([4.0, 1.5, 1.0, 1.0])
        t = np.array([target["calories"], target["protein"], target["carbs"], target["fat"]], float)
        A = M.T
        hi = np.full(A.shape[1], MAX_PORTION) if max_p is None else np.asarray(max_p, float)
        x = _bounded_lstsq(A / t[:, None] * w[:, None], w, MIN_PORTION, hi)
        x = np.clip(np.round(x * 4) / 4, MIN_PORTION, hi)          # quarter servings
        got = A @ x
        return x, got, float((w * ((got - t) / t) ** 2).sum())

    def _plan_meal(self, df, macros, slots, target, q, rng, trials=60):
        pools = {}
        for s in slots:
            idx = np.flatnonzero(df["slot_list"].apply(lambda l: s in l).to_numpy())
            if len(idx):
                pools[s] = idx
        if not pools:
            return None
        best = None
        for _ in range(trials):
            chosen, caps = [], []
            for slot, idx in pools.items():
                cand = idx[~np.isin(idx, chosen)]
                if cand.size == 0:
                    continue
                chosen.append(int(rng.choice(cand, p=_softmax(q[cand], 0.5))))
                caps.append(SLOT_MAX.get(slot, MAX_PORTION))
            x, got, err = self._fit(macros[chosen], target, caps)
            score = err - 0.05 * float(q[chosen].mean())
            if best is None or score < best[0]:
                best = (score, chosen, x)
        _, chosen, x = best
        rows = []
        for i, n in zip(chosen, x):
            f = df.iloc[i]
            rows.append({"food": f["name"], "serving": f["serving"], "servings": float(n),
                         "calories": round(f["calories"] * n), "protein_g": round(f["protein_g"] * n, 1),
                         "carbs_g": round(f["carbs_g"] * n, 1), "fat_g": round(f["fat_g"] * n, 1),
                         "fiber_g": round(f["fiber_g"] * n, 1), "sodium_mg": round(f["sodium_mg"] * n)})
        return rows

    @staticmethod
    def _sum(rows):
        return {k: round(sum(r[k] for r in rows), 1) for k in
                ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sodium_mg")}

    # ---------- public API ----------
    def generate(self, p: Profile, days: int = 7, seed: int = 0, strategy: str | None = None) -> dict:
        p.validate()
        strategy = strategy or predict_strategy(p)
        targets = compute_targets(p, strategy)
        df = self.eligible(p)
        if df.empty:
            raise ValueError("No foods match your restrictions.")
        pref = self._preference_bonus(df, p.likes)
        macros = df[["calories", "protein_g", "carbs_g", "fat_g"]].to_numpy(float)
        rng = np.random.default_rng(seed)
        usage, plan_days = {}, []

        for d in range(1, days + 1):
            remaining = {"calories": targets["calories"], "protein": targets["protein"],
                         "carbs": targets["carbs"], "fat": targets["fat"]}
            rem_share, meals = 1.0, []
            for name, share, slots in MEALS:
                frac = share / rem_share
                mt = {k: max(remaining[k] * frac, 0.3 * targets[k] * share) for k in remaining}
                q = self._quality(df, p, pref, usage).to_numpy()
                rows = self._plan_meal(df, macros, slots, mt, q, rng)
                if not rows:
                    continue
                tot = self._sum(rows)
                remaining["calories"] -= tot["calories"]; remaining["protein"] -= tot["protein_g"]
                remaining["carbs"] -= tot["carbs_g"];     remaining["fat"] -= tot["fat_g"]
                rem_share -= share
                for r in rows:
                    usage[r["food"]] = usage.get(r["food"], 0) + 1
                meals.append({"meal": name, "items": rows, "totals": tot})
            day_tot = self._sum([r for m in meals for r in m["items"]])
            plan_days.append({"day": d, "meals": meals, "totals": day_tot})

        avg = np.mean([d["totals"]["calories"] for d in plan_days])
        acc = round(100 * (1 - abs(avg - targets["calories"]) / targets["calories"]), 1)
        if acc < 95:
            targets["warnings"].append(
                "Your restrictions limit the food pool, so the plan is a bit off your calorie target. "
                "Relaxing a restriction or dislike will improve accuracy.")
        return {"profile": p, "targets": targets, "days": plan_days,
                "avg_calories": round(float(avg)),
                "calorie_accuracy_pct": acc,
                "foods_available": len(df)}
