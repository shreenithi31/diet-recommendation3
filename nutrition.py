"""Nutrition science: profile validation, BMR/TDEE, calorie and macro targets."""
from dataclasses import dataclass, field

ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55,
                    "active": 1.725, "very_active": 1.9}
GOALS = ("lose", "maintain", "gain")
DIETS = {"vegan": 0, "vegetarian": 1, "eggetarian": 2, "non-vegetarian": 3}
CONDITIONS = ("diabetes", "hypertension", "high_cholesterol")
ALLERGENS = ("dairy", "gluten", "nuts", "soy", "egg", "fish", "shellfish", "sesame")

# Macro split as fractions of calories
STRATEGIES = {
    "balanced":      {"carbs": 0.50, "protein": 0.20, "fat": 0.30,
                      "about": "Balanced split suited to general health."},
    "high_protein":  {"carbs": 0.40, "protein": 0.30, "fat": 0.30,
                      "about": "Higher protein to support muscle and fullness."},
    "low_carb":      {"carbs": 0.35, "protein": 0.30, "fat": 0.35,
                      "about": "Moderate carbs to help manage blood sugar."},
    "heart_healthy": {"carbs": 0.55, "protein": 0.20, "fat": 0.25,
                      "about": "Lower fat and sodium for heart health."},
}


@dataclass
class Profile:
    age: int
    sex: str
    weight_kg: float
    height_cm: float
    activity: str = "moderate"
    goal: str = "maintain"
    diet: str = "non-vegetarian"
    allergies: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    likes: list = field(default_factory=list)
    dislikes: list = field(default_factory=list)

    def validate(self):
        self.sex = self.sex.lower()
        if self.sex not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        if self.activity not in ACTIVITY_FACTORS:
            raise ValueError(f"activity must be one of {list(ACTIVITY_FACTORS)}")
        if self.goal not in GOALS:
            raise ValueError(f"goal must be one of {GOALS}")
        if self.diet not in DIETS:
            raise ValueError(f"diet must be one of {list(DIETS)}")
        if not 10 <= self.age <= 90:
            raise ValueError("age must be between 10 and 90")
        if not 30 <= self.weight_kg <= 250:
            raise ValueError("weight_kg must be between 30 and 250")
        if not 120 <= self.height_cm <= 230:
            raise ValueError("height_cm must be between 120 and 230")
        self.allergies = [a.lower().strip() for a in self.allergies if a.strip()]
        self.conditions = [c.lower().strip() for c in self.conditions if c.strip()]
        for a in self.allergies:
            if a not in ALLERGENS:
                raise ValueError(f"unknown allergy '{a}'. Choose from {ALLERGENS}")
        for c in self.conditions:
            if c not in CONDITIONS:
                raise ValueError(f"unknown condition '{c}'. Choose from {CONDITIONS}")
        return self


def bmi(weight_kg, height_cm):
    return weight_kg / (height_cm / 100) ** 2


def bmi_category(value):
    if value < 18.5: return "Underweight"
    if value < 25: return "Normal"
    if value < 30: return "Overweight"
    return "Obese"


def bmr(sex, weight_kg, height_cm, age):
    """Mifflin-St Jeor equation (kcal/day)."""
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161)


def rule_strategy(age, bmi_val, activity_idx, goal_idx, diabetes, hypertension, cholesterol):
    """Guideline-based policy used to label training data (and as a fallback)."""
    if diabetes:
        return "low_carb"
    if hypertension or cholesterol:
        return "heart_healthy"
    if goal_idx == 2 or (goal_idx == 0 and bmi_val >= 25) or activity_idx >= 3 or age >= 60:
        return "high_protein"
    return "balanced"


def compute_targets(p: Profile, strategy: str = "balanced") -> dict:
    p.validate()
    b = bmr(p.sex, p.weight_kg, p.height_cm, p.age)
    tdee = b * ACTIVITY_FACTORS[p.activity]
    bm = bmi(p.weight_kg, p.height_cm)
    goal, warnings = p.goal, []

    # Safety guard-rails
    if p.age < 18 and goal != "maintain":
        goal = "maintain"
        warnings.append("Under 18: plan set to maintenance. Please consult a pediatrician or dietitian "
                        "before any weight change.")
    if goal == "lose" and bm < 18.5:
        goal = "maintain"
        warnings.append("BMI is in the underweight range: plan set to maintenance, not weight loss.")
    if goal == "lose" and bm < 21:
        warnings.append("Your BMI is already low-normal; confirm a weight-loss goal with a professional.")

    if goal == "lose":
        calories = tdee - min(500, 0.20 * tdee)
        floor = 1500 if p.sex == "male" else 1200
        if calories < floor:
            calories = floor
            warnings.append(f"Calories raised to the safe minimum of {floor} kcal/day.")
    elif goal == "gain":
        calories = tdee + 300
    else:
        calories = tdee

    s = STRATEGIES[strategy]
    return {
        "bmr": round(b), "tdee": round(tdee), "bmi": round(bm, 1), "bmi_category": bmi_category(bm),
        "effective_goal": goal, "strategy": strategy, "strategy_about": s["about"],
        "calories": round(calories),
        "protein": round(calories * s["protein"] / 4),
        "carbs": round(calories * s["carbs"] / 4),
        "fat": round(calories * s["fat"] / 9),
        "fiber": round(14 * calories / 1000),
        "warnings": warnings,
    }
