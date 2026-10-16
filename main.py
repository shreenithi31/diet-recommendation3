"""CLI:  python src/main.py --age 28 --sex female --weight 62 --height 165 --goal lose --diet vegetarian"""
import argparse
from pathlib import Path
from nutrition import ACTIVITY_FACTORS, DIETS, GOALS, Profile
from recommender import DietRecommender
from report import format_markdown, format_text, shopping_list, to_dataframe


def csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="AI-based personalised diet recommendation")
    ap.add_argument("--age", type=int, default=30)
    ap.add_argument("--sex", default="male", choices=["male", "female"])
    ap.add_argument("--weight", type=float, default=75, help="kg")
    ap.add_argument("--height", type=float, default=172, help="cm")
    ap.add_argument("--activity", default="moderate", choices=list(ACTIVITY_FACTORS))
    ap.add_argument("--goal", default="maintain", choices=GOALS)
    ap.add_argument("--diet", default="non-vegetarian", choices=list(DIETS))
    ap.add_argument("--allergies", default="", help="comma list: dairy,gluten,nuts,soy,egg,fish,shellfish,sesame")
    ap.add_argument("--conditions", default="", help="comma list: diabetes,hypertension,high_cholesterol")
    ap.add_argument("--likes", default="", help="comma list of foods you enjoy")
    ap.add_argument("--dislikes", default="", help="comma list of foods to avoid")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", help="save to .csv, .json or .md")
    a = ap.parse_args()

    p = Profile(a.age, a.sex, a.weight, a.height, a.activity, a.goal, a.diet,
                csv_list(a.allergies), csv_list(a.conditions), csv_list(a.likes), csv_list(a.dislikes))
    plan = DietRecommender().generate(p, days=a.days, seed=a.seed)
    print(format_text(plan))

    if a.export:
        out = Path(a.export)
        if out.suffix == ".csv":
            to_dataframe(plan).to_csv(out, index=False)
        elif out.suffix == ".json":
            to_dataframe(plan).to_json(out, orient="records", indent=2)
        elif out.suffix == ".md":
            out.write_text(format_markdown(plan))
        else:
            raise SystemExit("Export must end in .csv, .json or .md")
        print(f"\nSaved plan to {out}")
        print("\nShopping list:\n", shopping_list(plan).to_string(index=False))


if __name__ == "__main__":
    main()
