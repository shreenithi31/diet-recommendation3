import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nutrition import Profile, bmr, compute_targets
from recommender import DietRecommender

REC = DietRecommender()


def make(**kw):
    base = dict(age=30, sex="male", weight_kg=75, height_cm=175, activity="moderate", goal="maintain")
    base.update(kw)
    return Profile(**base)


def foods_in(plan):
    return {r["food"] for d in plan["days"] for m in d["meals"] for r in m["items"]}


def test_bmr_mifflin():
    assert round(bmr("male", 80, 180, 30)) == 1780
    assert round(bmr("female", 60, 165, 25)) == 1345


def test_weight_loss_has_deficit_and_floor():
    t = compute_targets(make(goal="lose"))
    assert t["calories"] < t["tdee"]
    t = compute_targets(make(sex="female", weight_kg=48, height_cm=155, age=60, activity="sedentary", goal="lose"))
    assert t["calories"] >= 1200


def test_underweight_and_minor_cannot_diet():
    assert compute_targets(make(weight_kg=50, height_cm=178, goal="lose"))["effective_goal"] == "maintain"
    assert compute_targets(make(age=16, goal="lose"))["effective_goal"] == "maintain"


@pytest.mark.parametrize("diet,banned", [("vegan", {"nonveg", "egg", "veg"}),
                                         ("vegetarian", {"nonveg", "egg"}),
                                         ("eggetarian", {"nonveg"})])
def test_diet_respected(diet, banned):
    plan = REC.generate(make(diet=diet), days=5)
    lookup = REC.foods.set_index("name")["diet"]
    assert not {lookup[f] for f in foods_in(plan)} & banned


def test_allergies_never_appear():
    plan = REC.generate(make(allergies=["dairy", "nuts", "gluten", "soy"]), days=7)
    lookup = REC.foods.set_index("name")["allergen_set"]
    for f in foods_in(plan):
        assert not lookup[f] & {"dairy", "nuts", "gluten", "soy"}


def test_conditions_respected():
    plan = REC.generate(make(conditions=["diabetes", "hypertension"]), days=5)
    df = REC.foods.set_index("name")
    for f in foods_in(plan):
        assert df.loc[f, "gi"] != "high" and df.loc[f, "sodium_mg"] <= 400


def test_dislikes_excluded():
    assert not any("paneer" in f.lower() for f in foods_in(REC.generate(make(dislikes=["paneer"]), days=5)))


@pytest.mark.parametrize("kw", [dict(), dict(goal="lose", sex="female", weight_kg=70, height_cm=160),
                                dict(goal="gain", activity="active"), dict(diet="vegan", conditions=["diabetes"])])
def test_calories_close_to_target(kw):
    plan = REC.generate(make(**kw), days=7, seed=3)
    restricted = bool(kw.get("conditions"))
    assert plan["calorie_accuracy_pct"] >= (90 if restricted else 95)
    for d in plan["days"]:
        assert abs(d["totals"]["calories"] - plan["targets"]["calories"]) / plan["targets"]["calories"] < 0.15


def test_variety_across_week():
    plan = REC.generate(make(), days=7)
    dinners = [tuple(sorted(r["food"] for r in d["meals"][-1]["items"])) for d in plan["days"]]
    assert len(set(dinners)) >= 5


def test_reproducible_with_seed():
    a = REC.generate(make(), days=3, seed=11)
    b = REC.generate(make(), days=3, seed=11)
    assert a["days"] == b["days"]
