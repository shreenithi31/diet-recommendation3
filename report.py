"""Formatting helpers: dataframe, shopping list, text and markdown reports."""
import pandas as pd

DISCLAIMER = ("This tool gives general nutrition guidance, not medical advice. "
              "Consult a doctor or registered dietitian, especially with a medical condition, "
              "pregnancy, an eating disorder history, or before major diet changes.")


def to_dataframe(plan: dict) -> pd.DataFrame:
    rows = []
    for d in plan["days"]:
        for m in d["meals"]:
            for r in m["items"]:
                rows.append({"day": d["day"], "meal": m["meal"], "food": r["food"],
                             "amount": f'{r["servings"]:g} x {r["serving"]}', "calories": r["calories"],
                             "protein_g": r["protein_g"], "carbs_g": r["carbs_g"],
                             "fat_g": r["fat_g"], "fiber_g": r["fiber_g"]})
    return pd.DataFrame(rows)


def shopping_list(plan: dict) -> pd.DataFrame:
    agg = {}
    for d in plan["days"]:
        for m in d["meals"]:
            for r in m["items"]:
                e = agg.setdefault(r["food"], {"serving": r["serving"], "total_servings": 0.0})
                e["total_servings"] += r["servings"]
    out = pd.DataFrame([{"food": k, "total_servings": v["total_servings"], "serving_size": v["serving"]}
                        for k, v in agg.items()])
    return out.sort_values("food").reset_index(drop=True)


def format_text(plan: dict) -> str:
    t, L = plan["targets"], []
    L += ["=" * 64, "PERSONALISED DIET PLAN", "=" * 64,
          f"BMI {t['bmi']} ({t['bmi_category']}) | BMR {t['bmr']} | TDEE {t['tdee']} kcal",
          f"Strategy (AI-selected): {t['strategy']} - {t['strategy_about']}",
          f"Daily target: {t['calories']} kcal | Protein {t['protein']} g | Carbs {t['carbs']} g | "
          f"Fat {t['fat']} g | Fibre {t['fiber']} g"]
    for w in t["warnings"]:
        L.append(f"! {w}")
    for d in plan["days"]:
        L += ["", f"--- Day {d['day']} " + "-" * 50]
        for m in d["meals"]:
            L.append(f"{m['meal']} ({m['totals']['calories']:.0f} kcal)")
            for r in m["items"]:
                L.append(f"   - {r['food']:<34} {r['servings']:g} x {r['serving']:<18} {r['calories']:>4} kcal")
        x = d["totals"]
        L.append(f"Day total: {x['calories']:.0f} kcal | P {x['protein_g']} g | C {x['carbs_g']} g | "
                 f"F {x['fat_g']} g | Fibre {x['fiber_g']} g")
    L += ["", f"Average {plan['avg_calories']} kcal/day vs target {t['calories']} "
              f"({plan['calorie_accuracy_pct']}% match)", "", DISCLAIMER]
    return "\n".join(L)


def format_markdown(plan: dict) -> str:
    t, L = plan["targets"], ["# Personalised Diet Plan", ""]
    L += [f"**Target:** {t['calories']} kcal | P {t['protein']} g | C {t['carbs']} g | F {t['fat']} g "
          f"| Strategy: *{t['strategy']}*", ""]
    for d in plan["days"]:
        L += [f"## Day {d['day']}", "", "| Meal | Food | Amount | kcal |", "|---|---|---|---|"]
        for m in d["meals"]:
            for r in m["items"]:
                L.append(f"| {m['meal']} | {r['food']} | {r['servings']:g} x {r['serving']} | {r['calories']} |")
        L += ["", f"**Total:** {d['totals']['calories']:.0f} kcal", ""]
    L += [f"> {DISCLAIMER}"]
    return "\n".join(L)
