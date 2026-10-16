"""Content-based, preference-aware meal planner with portion optimisation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from nutrition import DIETS, Profile, compute_targets
from profile_model import predict_strategy


BASE_DIR = Path(__file__).resolve().parent

DIET_LEVEL = {
    "vegan": 0,
    "veg": 1,
    "egg": 2,
    "nonveg": 3,
}

MEALS = [
    ("Breakfast", 0.25, ["breakfast", "fruit", "dairy"]),
    ("Lunch", 0.35, ["protein", "carb", "veg"]),
    ("Snack", 0.10, ["snack", "fruit"]),
    ("Dinner", 0.30, ["protein", "carb", "veg"]),
]

MIN_PORTION, MAX_PORTION = 0.5, 2.5

SLOT_MAX = {
    "fruit": 2.0,
    "veg": 2.0,
    "dairy": 2.0,
    "snack": 2.0,
}


def load_foods(path=None) -> pd.DataFrame:
    """Load the food dataset from common project locations."""

    if path is None:
        candidates = [
            BASE_DIR / "foods.csv",
            BASE_DIR / "data" / "foods.csv",
            BASE_DIR / "src" / "data" / "foods.csv",
            BASE_DIR.parent / "foods.csv",
            BASE_DIR.parent / "data" / "foods.csv",
        ]

        path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )

    if path is None or not Path(path).is_file():
        searched = [
            BASE_DIR / "foods.csv",
            BASE_DIR / "data" / "foods.csv",
            BASE_DIR / "src" / "data" / "foods.csv",
            BASE_DIR.parent / "foods.csv",
            BASE_DIR.parent / "data" / "foods.csv",
        ]
        raise FileNotFoundError(
            "foods.csv was not found. Searched these locations:\n"
            + "\n".join(str(p) for p in searched)
            + "\nUpload foods.csv to your GitHub repository."
        )

    df = pd.read_csv(path)

    required = [
        "name", "allergens", "slots", "diet", "gi",
        "sodium_mg", "fat_g", "calories", "protein_g",
        "carbs_g", "fiber_g", "sugar_g", "serving",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"foods.csv is missing required columns: {missing}"
        )

    df["allergens"] = df["allergens"].fillna("")
    df["slots"] = df["slots"].fillna("")
    df["diet"] = df["diet"].fillna("").str.lower().str.strip()
    df["name"] = df["name"].fillna("")
    df["gi"] = df["gi"].fillna("").str.lower().str.strip()

    df["slot_list"] = df["slots"].str.split("|")
    df["allergen_set"] = df["allergens"].apply(
        lambda value: {
            item.strip().lower()
            for item in value.split(";")
            if item.strip()
        }
    )
    df["diet_level"] = df["diet"].map(DIET_LEVEL)

    df = df.dropna(subset=["diet_level"]).reset_index(drop=True)
    df["diet_level"] = df["diet_level"].astype(int)

    numeric_columns = [
        "calories", "protein_g", "carbs_g", "fat_g",
        "fiber_g", "sugar_g", "sodium_mg",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns)
    df = df[df["calories"] > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("foods.csv contains no valid food records.")

    return df


def _bounded_lstsq(A, b, lo, hi):
    """Solve bounded least squares with iterative clamping."""

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    fixed = np.zeros(len(x), dtype=bool)

    for _ in range(4):
        bad = ((x < lo) | (x > hi)) & ~fixed

        if not bad.any():
            break

        x[bad] = np.clip(x[bad], lo, hi[bad])
        fixed |= bad

        free = ~fixed

        if not free.any():
            break

        x[free] = np.linalg.lstsq(
            A[:, free],
            b - A[:, fixed] @ x[fixed],
            rcond=None,
        )[0]

    return np.clip(x, lo, hi)


def _softmax(x, tau):
    z = (np.asarray(x, dtype=float) - np.max(x)) / tau
    e = np.exp(z)
    return e / e.sum()


class DietRecommender:

    def __init__(self, foods: pd.DataFrame | None = None):
        self.foods = load_foods() if foods is None else foods

    # ---------- Filtering ----------

    def eligible(self, p: Profile) -> pd.DataFrame:
        df = self.foods

        mask = df["diet_level"] <= DIETS[p.diet]

        allergies = {
            allergy.strip().lower()
            for allergy in p.allergies
        }

        mask &= df["allergen_set"].apply(
            lambda food_allergens: not (food_allergens & allergies)
        )

        dislikes = [
            dislike.lower().strip()
            for dislike in p.dislikes
            if dislike.strip()
        ]

        names = df["name"].str.lower()

        if dislikes:
            mask &= ~names.apply(
                lambda name: any(dislike in name for dislike in dislikes)
            )

        if "diabetes" in p.conditions:
            mask &= df["gi"] != "high"

        if "hypertension" in p.conditions:
            mask &= df["sodium_mg"] <= 400

        if "high_cholesterol" in p.conditions:
            mask &= df["fat_g"] <= 14

        return df[mask].reset_index(drop=True)

    # ---------- Preference scoring ----------

    @staticmethod
    def _preference_bonus(df: pd.DataFrame, likes) -> pd.Series:
        bonus = pd.Series(0.0, index=df.index)

        likes = [
            like.lower().strip()
            for like in likes
            if like.strip()
        ]

        if not likes or df.empty:
            return bonus

        liked = df["name"].str.lower().apply(
            lambda name: any(like in name for like in likes)
        )

        if not liked.any():
            return bonus

        dens = df[
            ["protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g"]
        ].div(df["calories"] / 100, axis=0)

        X = StandardScaler().fit_transform(dens)

        sim = cosine_similarity(
            X, X[liked.to_numpy()]
        ).max(axis=1)

        return (
            0.5 * pd.Series(np.clip(sim, 0, 1), index=df.index)
            + liked.astype(float)
        )

    @staticmethod
    def _quality(df, p, pref, usage) -> pd.Series:
        fiber_density = (
            df["fiber_g"] / (df["calories"] / 100)
        ).clip(upper=6)

        sodium_weight = (
            1.5 if "hypertension" in p.conditions else 0.5
        )

        sugar_weight = (
            2.0 if "diabetes" in p.conditions else 1.0
        )

        quality = (
            1
            + 0.12 * fiber_density
            - sodium_weight * df["sodium_mg"] / 1000
            - sugar_weight * df["sugar_g"] / 50
        )

        return quality + pref - 0.7 * df["name"].map(usage).fillna(0)

    # ---------- Portion optimisation ----------

    @staticmethod
    def _fit(M: np.ndarray, target: dict, max_p=None):
        """Fit portions to calories, protein, carbohydrates and fat."""

        weights = np.array([4.0, 1.5, 1.0, 1.0])

        target_values = np.array(
            [
                target["calories"],
                target["protein"],
                target["carbs"],
                target["fat"],
            ],
            dtype=float,
        )

        target_values = np.maximum(target_values, 1e-6)

        A = M.T

        upper = (
            np.full(A.shape[1], MAX_PORTION)
            if max_p is None
            else np.asarray(max_p, dtype=float)
        )

        x = _bounded_lstsq(
            A / target_values[:, None] * weights[:, None],
            weights,
            MIN_PORTION,
            upper,
        )

        x = np.clip(np.round(x * 4) / 4, MIN_PORTION, upper)

        achieved = A @ x

        error = float(
            (
                weights
                * ((achieved - target_values) / target_values) ** 2
            ).sum()
        )

        return x, achieved, error

    def _plan_meal(self, df, macros, slots, target, q, rng, trials=60):
        pools = {}

        for slot in slots:
            indices = np.flatnonzero(
                df["slot_list"].apply(
                    lambda slot_list: slot in slot_list
                ).to_numpy()
            )

            if len(indices):
                pools[slot] = indices

        if not pools:
            return None

        best = None

        for _ in range(trials):
            chosen = []
            caps = []

            for slot, indices in pools.items():
                candidates = indices[
                    ~np.isin(indices, chosen)
                ]

                if candidates.size == 0:
                    continue

                chosen.append(
                    int(
                        rng.choice(
                            candidates,
                            p=_softmax(q[candidates], 0.5),
                        )
                    )
                )

                caps.append(SLOT_MAX.get(slot, MAX_PORTION))

            if not chosen:
                continue

            portions, _, error = self._fit(
                macros[chosen],
                target,
                caps,
            )

            score = error - 0.05 * float(q[chosen].mean())

            if best is None or score < best[0]:
                best = (score, chosen, portions)

        if best is None:
            return None

        _, chosen, portions = best
        rows = []

        for index, portion in zip(chosen, portions):
            food = df.iloc[index]

            rows.append(
                {
                    "food": food["name"],
                    "serving": food["serving"],
                    "servings": float(portion),
                    "calories": round(food["calories"] * portion),
                    "protein_g": round(food["protein_g"] * portion, 1),
                    "carbs_g": round(food["carbs_g"] * portion, 1),
                    "fat_g": round(food["fat_g"] * portion, 1),
                    "fiber_g": round(food["fiber_g"] * portion, 1),
                    "sodium_mg": round(food["sodium_mg"] * portion),
                }
            )

        return rows

    @staticmethod
    def _sum(rows):
        return {
            key: round(sum(row[key] for row in rows), 1)
            for key in (
                "calories", "protein_g", "carbs_g",
                "fat_g", "fiber_g", "sodium_mg",
            )
        }

    # ---------- Public API ----------

    def generate(
        self,
        p: Profile,
        days: int = 7,
        seed: int = 0,
        strategy: str | None = None,
    ) -> dict:

        p.validate()

        strategy = strategy or predict_strategy(p)
        targets = compute_targets(p, strategy)

        df = self.eligible(p)

        if df.empty:
            raise ValueError("No foods match your restrictions.")

        pref = self._preference_bonus(df, p.likes)

        macros = df[
            ["calories", "protein_g", "carbs_g", "fat_g"]
        ].to_numpy(dtype=float)

        rng = np.random.default_rng(seed)

        usage = {}
        plan_days = []

        for day in range(1, days + 1):

            remaining = {
                "calories": targets["calories"],
                "protein": targets["protein"],
                "carbs": targets["carbs"],
                "fat": targets["fat"],
            }

            remaining_share = 1.0
            meals = []

            for meal_name, share, slots in MEALS:

                fraction = share / remaining_share

                meal_target = {
                    key: max(
                        remaining[key] * fraction,
                        0.3 * targets[key] * share,
                    )
                    for key in remaining
                }

                quality = self._quality(
                    df, p, pref, usage
                ).to_numpy()

                rows = self._plan_meal(
                    df, macros, slots, meal_target,
                    quality, rng,
                )

                if not rows:
                    continue

                totals = self._sum(rows)

                remaining["calories"] -= totals["calories"]
                remaining["protein"] -= totals["protein_g"]
                remaining["carbs"] -= totals["carbs_g"]
                remaining["fat"] -= totals["fat_g"]

                remaining_share -= share

                for row in rows:
                    food_name = row["food"]
                    usage[food_name] = usage.get(food_name, 0) + 1

                meals.append(
                    {
                        "meal": meal_name,
                        "items": rows,
                        "totals": totals,
                    }
                )

            day_totals = self._sum(
                [
                    row
                    for meal in meals
                    for row in meal["items"]
                ]
            )

            plan_days.append(
                {
                    "day": day,
                    "meals": meals,
                    "totals": day_totals,
                }
            )

        average_calories = float(
            np.mean(
                [day["totals"]["calories"] for day in plan_days]
            )
        )

        accuracy = round(
            100 * (
                1
                - abs(average_calories - targets["calories"])
                / max(targets["calories"], 1)
            ),
            1,
        )

        if accuracy < 95:
            targets["warnings"].append(
                "Your restrictions limit the food pool, so the plan "
                "is a bit off your calorie target. Relaxing a "
                "restriction or dislike may improve accuracy."
            )

        return {
            "profile": p,
            "targets": targets,
            "days": plan_days,
            "avg_calories": round(average_calories),
            "calorie_accuracy_pct": accuracy,
            "foods_available": len(df),
        }

