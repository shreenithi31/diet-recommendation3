# 🥗 AI-Based Personalised Diet Recommendation

Generates a personalised multi-day meal plan from a person's body metrics, goal, dietary
preference, allergies, health conditions and food likes/dislikes. Includes a CLI, a Streamlit
web app, an ML model, and automated tests.

## How it works
| Stage | Technique |
|---|---|
| 1. Energy needs | Mifflin-St Jeor BMR × activity factor = TDEE; goal adjusts calories (deficit ≤ 20 %, surplus +300) |
| 2. Macro strategy | **Random Forest classifier** picks `balanced / high_protein / low_carb / heart_healthy` from age, BMI, sex, activity, goal and conditions |
| 3. Safe filtering | Hard filters for diet type, allergens, dislikes, and condition rules (diabetes → no high-GI; hypertension → sodium ≤ 400 mg/item; high cholesterol → fat ≤ 14 g/item) |
| 4. Ranking | **Content-based filtering**: nutrient-density quality score (fiber, sodium, sugar) + cosine similarity to foods you like + repeat penalty for weekly variety |
| 5. Meal building | Per meal, 60 randomised food combinations are scored; portions of each are solved by **bounded least squares** to hit calorie/macro targets (quarter-serving steps) |
| 6. Drift correction | Each meal target is recomputed from what's left of the day's budget |

### Safety guard-rails
- Weight loss is never prescribed for **under-18s** or **BMI < 18.5** (plan falls back to maintenance)
- Calorie floor of 1,500 (men) / 1,200 (women) for weight loss; deficit capped at 20 % of TDEE
- Warnings when restrictions make the target hard to hit
- Disclaimer in every output. This is general guidance, **not medical advice**

> **Note on the ML model:** it is trained on synthetic profiles labelled by an encoded dietary-guideline
> policy (+4 % noise). For production, retrain on dietitian-labelled data in `src/profile_model.py`.

## Project structure
```
diet-recommendation/
├── app.py                  # Streamlit web app
├── data/foods.csv          # 81 foods with nutrition, allergens, GI, diet type (edit/extend freely)
├── models/                 # profile_model.joblib (generated)
├── src/
│   ├── nutrition.py        # Profile, BMR/TDEE, targets, safety rules
│   ├── profile_model.py    # ML macro-strategy classifier (train + predict)
│   ├── recommender.py      # filtering, ranking, portion optimisation, planner
│   ├── report.py           # text/markdown/CSV output + shopping list
│   └── main.py             # command-line interface
└── tests/test_diet.py      # 15 tests (diet, allergies, conditions, calorie accuracy, safety)
```

## Quick start
```bash
pip install -r requirements.txt
python src/profile_model.py          # train the strategy model (optional; falls back to rules)
streamlit run app.py                 # web app
python -m pytest -q tests            # run tests
```

### CLI examples
```bash
python src/main.py --age 28 --sex female --weight 62 --height 165 \
    --goal lose --diet vegetarian --activity light --days 3 --likes "paneer,oat" --dislikes mushroom

python src/main.py --age 45 --sex male --weight 88 --height 175 --goal lose \
    --conditions hypertension --allergies shellfish --days 7 --export plan.csv
```
Export formats: `.csv`, `.json`, `.md` (a shopping list is printed with exports).

### Use as a library
```python
import sys; sys.path.insert(0, "src")
from nutrition import Profile
from recommender import DietRecommender

p = Profile(age=30, sex="male", weight_kg=75, height_cm=175, goal="gain", diet="vegetarian")
plan = DietRecommender().generate(p, days=7, seed=1)
print(plan["targets"], plan["avg_calories"])
```

## Extending
- Add foods to `data/foods.csv` (columns documented by existing rows; `slots` = breakfast, protein, carb, veg, fruit, dairy, snack)
- Add regional cuisines, micronutrients (iron, B12, calcium) and use them in `_quality`
- Learn preferences from ratings (collaborative filtering) once you collect user feedback
- Swap in the USDA FoodData Central API for a larger food database
