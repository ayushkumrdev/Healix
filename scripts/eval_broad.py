#!/usr/bin/env python3
"""Broad-question subtopic coverage: does the RAPTOR summary tier cover more of a
multi-part answer than flat leaf retrieval, at a fixed context budget?

Motivation: broad questions ("how does diabetes affect the body") have no single
gold chunk; a good answer aggregates several subtopics. Flat retrieval fills the
context with narrow, often redundant leaf passages; RAPTOR overview nodes each
summarize a whole cluster, so they should pack more distinct subtopics into the
same budget. We measure this directly.

Pre-registered hypothesis (before running): at a fixed character budget B, the
RAPTOR overview tier (and the altitude-matched union) covers more of a question's
objective subtopics than flat leaf retrieval.

Method: for each hand-authored broad question with an objective subtopic list
(textbook medical aspects, not tuned to any method), fill budget B by
concatenating ranked retrieved text (truncating the last item). A subtopic counts
as covered if any of its synonym keywords appears in the budgeted text. We report
mean coverage per method and a paired bootstrap 95% CI on the
(RAPTOR-overview - flat-leaf) coverage gap. Fixing B controls for the fact that
summary nodes are longer than leaf chunks.

Usage:  .venv/Scripts/python.exe scripts/eval_broad.py --budget 2000
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
ROOT = Path(__file__).resolve().parent.parent

# Broad questions with objective subtopic keyword sets (textbook medical aspects).
# Each subtopic is a list of accepted synonyms; covered if any appears.
QUESTIONS = [
    {"q": "how does diabetes affect the body over time",
     "subs": [["retinopath", "eye", "vision", "retina"], ["nephropath", "kidney", "renal"],
              ["neuropath", "nerve"], ["cardiovascular", "heart", "coronary"],
              ["foot", "ulcer", "amputation"], ["wound", "healing", "infection"]]},
    {"q": "what are the effects of chronic stress on the body",
     "subs": [["cardiovascular", "blood pressure", "heart"], ["immune", "infection"],
              ["digest", "gut", "stomach", "gastro"], ["sleep", "insomnia"],
              ["anxiety", "depress", "mood"], ["cortisol", "hpa", "adrenal"]]},
    {"q": "what are the risk factors for heart disease",
     "subs": [["smok", "tobacco"], ["cholesterol", "lipid"], ["hypertension", "blood pressure"],
              ["diabet"], ["obes", "weight"], ["inactiv", "sedentary", "exercise"],
              ["family history", "genetic", "hereditary"]]},
    {"q": "how does high blood pressure damage the body",
     "subs": [["stroke", "brain"], ["kidney", "renal"], ["heart", "cardiac", "coronary"],
              ["eye", "retina", "vision"], ["artery", "arteri", "aneurysm"]]},
    {"q": "what are the complications of obesity",
     "subs": [["diabet"], ["heart", "cardiovascular"], ["hypertension", "blood pressure"],
              ["sleep apnea", "apnea"], ["joint", "arthritis", "osteoarthritis"],
              ["cancer"], ["liver", "fatty"]]},
    {"q": "what happens to the body during menopause",
     "subs": [["hot flash", "flush"], ["bone", "osteoporos"], ["mood", "depress", "irritab"],
              ["sleep"], ["vaginal", "dryness"], ["heart", "cardiovascular"]]},
    {"q": "how does smoking affect health",
     "subs": [["lung", "copd", "emphysema"], ["cancer"], ["heart", "cardiovascular", "coronary"],
              ["stroke"], ["pregnan", "fetal"], ["gum", "teeth", "oral", "dental"]]},
    {"q": "what are the symptoms and effects of hypothyroidism",
     "subs": [["fatigue", "tired"], ["weight gain"], ["cold", "temperature"],
              ["dry skin", "hair"], ["depress", "mood"], ["constipat"], ["heart", "brady"]]},
    {"q": "how does chronic kidney disease affect the body",
     "subs": [["anemia"], ["bone", "mineral"], ["blood pressure", "hypertension"],
              ["fluid", "edema", "swelling"], ["heart", "cardiovascular"],
              ["electrolyte", "potassium"]]},
    {"q": "what are the long term effects of alcohol on the body",
     "subs": [["liver", "cirrhosis", "hepat"], ["brain", "cognit", "memory"],
              ["heart", "cardiomyopath", "cardiovascular"], ["pancrea"],
              ["cancer"], ["depend", "addict"]]},
    {"q": "how does pregnancy change the body",
     "subs": [["blood volume", "cardiovascular", "heart"], ["hormone", "estrogen", "progesterone"],
              ["weight", "breast"], ["nausea", "morning sickness"], ["back", "joint", "ligament"],
              ["blood pressure", "gestational"]]},
    {"q": "what are the effects of sleep deprivation",
     "subs": [["cognit", "concentrat", "memory"], ["mood", "irritab", "depress"],
              ["immune"], ["weight", "appetite", "metabol"], ["heart", "cardiovascular", "blood pressure"],
              ["accident", "reaction"]]},
    {"q": "how does asthma affect the respiratory system",
     "subs": [["airway", "bronch"], ["inflammat"], ["wheez"], ["shortness", "breath", "dyspnea"],
              ["mucus", "phlegm"], ["trigger", "allerg"]]},
    {"q": "what are the complications of untreated depression",
     "subs": [["suicid", "self-harm"], ["substance", "alcohol", "drug"],
              ["heart", "cardiovascular"], ["relationship", "social", "isolat"],
              ["work", "function", "productiv"], ["sleep"]]},
    {"q": "how does anemia affect the body",
     "subs": [["fatigue", "tired", "weak"], ["short", "breath", "dyspnea"], ["pale", "pallor"],
              ["heart", "palpitat", "tachy"], ["dizz", "headache"], ["cold hand", "cold feet"]]},
    {"q": "what are the health effects of vitamin D deficiency",
     "subs": [["bone", "osteoporos", "osteomalacia", "rickets"], ["muscle", "weak"],
              ["immune"], ["mood", "depress"], ["fatigue"], ["fall", "fracture"]]},
    {"q": "how does COPD affect the body",
     "subs": [["airway", "bronch"], ["short", "breath", "dyspnea"], ["cough", "mucus", "sputum"],
              ["oxygen", "hypox"], ["heart", "cor pulmonale", "cardiovascular"], ["exacerbat", "infection"]]},
    {"q": "what are the effects of dehydration on the body",
     "subs": [["thirst", "dry mouth"], ["kidney", "urine"], ["blood pressure", "dizz"],
              ["fatigue", "confus"], ["heart", "palpitat", "rate"], ["headache"]]},
    {"q": "how does rheumatoid arthritis affect the body",
     "subs": [["joint", "synov"], ["inflammat"], ["stiffness"], ["fatigue"],
              ["heart", "cardiovascular", "lung"], ["deform", "erosion"]]},
    {"q": "what are the complications of high cholesterol",
     "subs": [["atheroscleros", "plaque", "artery"], ["heart", "coronary"], ["stroke"],
              ["peripheral", "leg", "claudicat"], ["pancrea"]]},
    {"q": "how does hyperthyroidism affect the body",
     "subs": [["weight loss"], ["heart", "palpitat", "tachy", "atrial"], ["anxiety", "nervous", "irritab"],
              ["heat", "sweat"], ["tremor", "shak"], ["eye", "exophthalmos", "goiter"]]},
    {"q": "what are the effects of iron deficiency",
     "subs": [["anemia"], ["fatigue", "weak"], ["pale", "pallor"], ["short", "breath"],
              ["brittle nail", "hair"], ["restless leg", "pica"]]},
    {"q": "how does liver cirrhosis affect the body",
     "subs": [["jaundice", "bilirubin"], ["ascites", "fluid", "swelling"], ["bleed", "varice", "clot"],
              ["encephalopath", "confus", "brain"], ["portal", "hypertension"], ["infection", "immune"]]},
    {"q": "what are the systemic effects of lupus",
     "subs": [["joint", "arthritis"], ["skin", "rash", "butterfly"], ["kidney", "nephritis", "renal"],
              ["heart", "pericard", "cardiovascular"], ["lung", "pleur"], ["fatigue"], ["blood", "anemia", "cytopenia"]]},
]


def covered(text, subs):
    t = text.lower()
    return sum(1 for syns in subs if any(s in t for s in syns))


def fill_budget(items, budget):
    """Concatenate item texts up to `budget` chars (truncate the last)."""
    out, used = [], 0
    for it in items:
        txt = (it.get("text") or "")
        if not txt:
            continue
        take = txt[: max(0, budget - used)]
        out.append(take)
        used += len(take)
        if used >= budget:
            break
    return " ".join(out)


def bootstrap_ci(a, b, iters=5000, seed=1):
    rng = random.Random(seed); n = len(a); diffs = []
    for _ in range(iters):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in s)/n - sum(b[i] for i in s)/n)
    diffs.sort()
    return (round(sum(a)/n - sum(b)/n, 3),
            round(diffs[int(0.025*iters)], 3), round(diffs[int(0.975*iters)], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--pool", type=int, default=12, help="items retrieved per method before budgeting")
    args = ap.parse_args()

    from services.retriever import MedicalRetriever
    from services.raptor import RaptorIndex
    retr = MedicalRetriever(index_dir=str(ROOT / "data"))
    retr.retrieve("warmup", k=3)
    rap = RaptorIndex()
    if not rap.available:
        print("RAPTOR index unavailable; build it first."); return

    per = {"flat_leaf": [], "raptor_overview": [], "altitude_union": []}
    total = len(QUESTIONS[0]["subs"])  # not used; per-question totals below
    frac = {k: [] for k in per}
    for item in QUESTIONS:
        q, subs = item["q"], item["subs"]
        nsub = len(subs)
        leaf = retr.hybrid_retrieve(query=q, k=args.pool)
        overview = rap.search(q, retr, k=args.pool, min_score=0.0)
        # altitude union: overview first (frame), then leaf (detail), dedup by text
        seen, union = set(), []
        for it in overview + leaf:
            key = (it.get("text") or "")[:60]
            if key and key not in seen:
                seen.add(key); union.append(it)

        cov_leaf = covered(fill_budget(leaf, args.budget), subs)
        cov_over = covered(fill_budget(overview, args.budget), subs)
        cov_uni = covered(fill_budget(union, args.budget), subs)
        per["flat_leaf"].append(cov_leaf)
        per["raptor_overview"].append(cov_over)
        per["altitude_union"].append(cov_uni)
        frac["flat_leaf"].append(cov_leaf / nsub)
        frac["raptor_overview"].append(cov_over / nsub)
        frac["altitude_union"].append(cov_uni / nsub)

    report = {"budget_chars": args.budget, "pool": args.pool, "n_questions": len(QUESTIONS),
              "mean_coverage_fraction": {k: round(sum(v)/len(v), 3) for k, v in frac.items()},
              "mean_subtopics_covered": {k: round(sum(v)/len(v), 2) for k, v in per.items()}}
    # significance: overview vs leaf, and union vs leaf (fraction)
    pt, lo, hi = bootstrap_ci(frac["raptor_overview"], frac["flat_leaf"])
    report["overview_minus_leaf"] = {"gap": pt, "ci95": [lo, hi], "significant": lo > 0 or hi < 0}
    pt2, lo2, hi2 = bootstrap_ci(frac["altitude_union"], frac["flat_leaf"])
    report["union_minus_leaf"] = {"gap": pt2, "ci95": [lo2, hi2], "significant": lo2 > 0 or hi2 < 0}

    out = ROOT / "data" / "eval_broad.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== RESULTS ===")
    print(json.dumps(report, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
