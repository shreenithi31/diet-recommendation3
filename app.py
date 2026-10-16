"""Streamlit web app:  streamlit run app.py"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from nutrition import ACTIVITY_FACTORS, ALLERGENS, CONDITIONS, DIETS, GOALS, Profile
from recommender import DietRecommender
from report import DISCLAIMER, format_markdown, shopping_list, to_dataframe

st.set_page_config(page_title="AI Diet Recommender", page_icon="🥗", layout="wide")


@st.cache_resource
def get_recommender():
    return DietRecommender()


rec = get_recommender()
food_names = sorted(rec.foods["name"])

st.title("🥗 AI-Based Personalised Diet Recommendation")
st.caption("Tell us about yourself and get a 7-day meal plan tuned to your body, goals, and tastes.")

with st.sidebar:
    st.header("Your profile")
    age = st.number_input("Age", 10, 90, 30)
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, 0.5)
    height = st.number_input("Height (cm)", 120.0, 230.0, 170.0, 0.5)
    activity = st.selectbox("Activity level", list(ACTIVITY_FACTORS), 2)
    goal = st.selectbox("Goal", GOALS, 1, format_func=lambda g: {"lose": "Lose weight", "maintain": "Maintain",
                                                                 "gain": "Gain weight / muscle"}[g])
    diet = st.selectbox("Diet type", list(DIETS), 3)
    allergies = st.multiselect("Allergies", ALLERGENS)
    conditions = st.multiselect("Health conditions", CONDITIONS)
    likes = st.multiselect("Foods you like", food_names)
    dislikes = st.multiselect("Foods to avoid", food_names)
    days = st.slider("Days", 1, 7, 7)
    seed = st.number_input("Variation seed", 0, 9999, 0, help="Change for a different plan")
    go = st.button("Generate plan", type="primary", width="stretch")

if not go:
    st.info("Fill in your profile in the sidebar and click **Generate plan**.")
    st.caption(DISCLAIMER)
    st.stop()

try:
    plan = rec.generate(Profile(age, sex, weight, height, activity, goal, diet,
                                allergies, conditions, likes, dislikes), days=days, seed=int(seed))
except ValueError as e:
    st.error(str(e)); st.stop()

t = plan["targets"]
for w in t["warnings"]:
    st.warning(w)
c = st.columns(5)
c[0].metric("BMI", t["bmi"], t["bmi_category"], delta_color="off")
c[1].metric("Daily calories", f"{t['calories']} kcal")
c[2].metric("Protein", f"{t['protein']} g")
c[3].metric("Carbs", f"{t['carbs']} g")
c[4].metric("Fat", f"{t['fat']} g")
st.success(f"AI-selected strategy: **{t['strategy'].replace('_', ' ')}** — {t['strategy_about']} "
           f"(plan matches your calorie target within {100 - plan['calorie_accuracy_pct']:.1f}%)")

tab_plan, tab_week, tab_shop = st.tabs(["📅 Meal plan", "📊 Weekly overview", "🛒 Shopping list"])

with tab_plan:
    for d, tab in zip(plan["days"], st.tabs([f"Day {d['day']}" for d in plan["days"]])):
        with tab:
            for m in d["meals"]:
                st.subheader(f"{m['meal']} · {m['totals']['calories']:.0f} kcal")
                st.dataframe(pd.DataFrame([{"Food": r["food"], "Amount": f"{r['servings']:g} × {r['serving']}",
                                            "kcal": r["calories"], "Protein g": r["protein_g"],
                                            "Carbs g": r["carbs_g"], "Fat g": r["fat_g"]} for r in m["items"]]),
                             hide_index=True, width="stretch")
            x = d["totals"]
            st.markdown(f"**Day total:** {x['calories']:.0f} kcal · P {x['protein_g']} g · "
                        f"C {x['carbs_g']} g · F {x['fat_g']} g · Fibre {x['fiber_g']} g")

with tab_week:
    wk = pd.DataFrame([{"Day": f"Day {d['day']}", "Calories": d["totals"]["calories"],
                        "Target": t["calories"]} for d in plan["days"]]).set_index("Day")
    st.bar_chart(wk["Calories"])
    st.caption(f"Average {plan['avg_calories']} kcal/day vs target {t['calories']} kcal.")
    macro = pd.DataFrame({"Planned (avg)": [sum(d["totals"][k] for d in plan["days"]) / len(plan["days"])
                                            for k in ("protein_g", "carbs_g", "fat_g")],
                          "Target": [t["protein"], t["carbs"], t["fat"]]},
                         index=["Protein g", "Carbs g", "Fat g"])
    st.bar_chart(macro)

with tab_shop:
    st.dataframe(shopping_list(plan), hide_index=True, width="stretch")

st.download_button("⬇️ Download plan (CSV)", to_dataframe(plan).to_csv(index=False), "diet_plan.csv", "text/csv")
st.download_button("⬇️ Download plan (Markdown)", format_markdown(plan), "diet_plan.md")
st.caption(DISCLAIMER)
