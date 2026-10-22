#!/usr/bin/env python
# /// script
# dependencies = ["anthropic>=2.1"]
# ///
"""
Generate synthetic negative headlines (not water conflict) with Claude Haiku 5.5.

Each call draws one combination from the diversity axes below (category ×
subtopic × regions × headline style) and asks for a batch of headlines. Real
Google News titles cached by fetch_serpapi_negatives.py are passed in as style
examples, so the output reads like real headlines rather than LLM prose.

Categories:
- conflict_no_water: violence, unrest, crime and security news with no water role
- water_no_conflict: water-related news with no violence (false-positive space)
- general:           everyday news with neither

Every call is cached in ../data/synthetic_raw/ by its index, so re-runs only
make the calls that are missing. Verify the output with verify_negatives.py
before using it: the generator can still drift into water conflict.

Output:
- ../data/synthetic_headlines.csv  (Headline, category, subtopic, regions, style, call)

Usage:
    uv run generate_synthetic_negatives.py --calls 2      # test run
    uv run generate_synthetic_negatives.py --calls 120    # ~3,000 headlines

Requirements:
    ANTHROPIC_API_KEY in ../.env (or the environment)
"""

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from negatives_common import (
    DATA_DIR,
    SERPAPI_RAW_DIR,
    SYNTHETIC_HEADLINES_PATH,
    WATER_CONFLICT_DEFINITION,
    load_env_key,
    normalize,
)

MODEL = "claude-haiku-5-5"
RAW_DIR = DATA_DIR / "synthetic_raw"
HEADLINES_PER_CALL = 25
SEEDS_PER_CALL = 8

# (category, share of calls)
CATEGORY_WEIGHTS = [
    ("conflict_no_water", 0.40),
    ("water_no_conflict", 0.30),
    ("general", 0.30),
]

SUBTOPICS = {
    "conflict_no_water": [
        "airstrikes and drone strikes on military targets, homes or markets",
        "attacks on power grids, fuel depots, railways, bridges or ports",
        "attacks on hospitals, schools or aid convoys",
        "armed group raids on villages, kidnappings and abductions",
        "insurgent ambushes and roadside bombs against soldiers or police",
        "police clashing with protesters over elections, fuel prices or wages",
        "riots over food prices, taxes, police killings or cost of living",
        "strikes and labour protests by teachers, doctors, miners or transport workers",
        "election violence, assassinations of politicians, attacks on polling stations",
        "ethnic or sectarian clashes, mob violence and lynchings",
        "gang and cartel violence, prison riots, organised crime shootings",
        "border skirmishes, shelling between armies, ceasefire violations",
        "military coups, mutinies and crackdowns on opposition",
        "land disputes over urban property, evictions and demolitions",
        "attacks on journalists, activists or religious minorities",
        "cyberattacks, sabotage of telecoms and arson attacks on government buildings",
        "student protests and university campus clashes",
        "piracy, maritime seizures and attacks on cargo ships at sea",
    ],
    "water_no_conflict": [
        "water utility tariffs, budgets, contracts and infrastructure upgrades",
        "dam, hydropower and canal construction milestones and inaugurations",
        "floods, storms and heavy rain as natural disasters, evacuations and relief",
        "drought impacts on crops, prices and livestock, without unrest",
        "water treatment, boil-water advisories, contamination tests and recalls",
        "water research: aquifers, glaciers, desalination, filtration technology",
        "transboundary water cooperation: treaties signed, joint commissions, data sharing",
        "court rulings, fines and regulations on pollution and water rights",
        "rivers, lakes and coasts: tourism, cleanups, wildlife and fishing seasons",
        "water sports: swimming, rowing, sailing, surfing, water polo results",
        "shipping and river transport: ports, ferries, canal traffic, low river levels",
        "idioms and names with water words: Watergate, watered-down bills, underwater mortgages, Waterloo, Riverside",
        "water access projects: new wells, pipelines and taps opened in villages",
        "water business: bottled water companies, utility mergers, water stocks",
    ],
    "general": [
        "company earnings, mergers, layoffs and stock markets",
        "technology launches, AI, smartphones and startups",
        "football, cricket, basketball, athletics and tennis results",
        "health: vaccines, hospital waiting lists, disease outbreaks, medical studies",
        "science and space: discoveries, launches, archaeology",
        "film, music, celebrities, festivals and awards",
        "education: exams, school funding, university admissions",
        "transport accidents: road crashes, train derailments, plane incidents",
        "courts and crime: fraud trials, sentencing, corruption charges",
        "politics: parliament votes, cabinet changes, election results, polls",
        "economy: inflation, interest rates, budgets, fuel and food prices",
        "diplomacy: summits, state visits, trade deals, sanctions",
        "weather other than water: heatwaves, cold snaps, wildfires, earthquakes",
        "local news: council decisions, roadworks, new markets, community events",
        "agriculture and food: harvests, crop prices, livestock markets, fertiliser",
        "migration and refugees: asylum policy, resettlement, remittances",
    ],
}

REGIONS = [
    "Nigeria", "Kenya", "Ethiopia", "Sudan", "South Sudan", "Somalia", "DR Congo", "Mali",
    "Burkina Faso", "Niger", "Cameroon", "South Africa", "Zimbabwe", "Mozambique", "Egypt",
    "Libya", "Morocco", "Ghana", "Uganda", "Tanzania", "Iraq", "Syria", "Yemen", "Lebanon",
    "Israel and Palestine", "Iran", "Jordan", "Turkey", "Afghanistan", "Pakistan", "India",
    "Bangladesh", "Nepal", "Sri Lanka", "Myanmar", "Thailand", "Philippines", "Indonesia",
    "China", "Central Asia", "Ukraine", "Russia", "Georgia and Armenia", "Balkans",
    "France", "Germany", "United Kingdom", "Spain", "Italy", "United States", "Mexico",
    "Guatemala and Honduras", "Haiti", "Colombia", "Venezuela", "Peru", "Bolivia", "Brazil",
    "Chile", "Argentina", "Australia", "Pacific Islands",
]

STYLES = [
    "terse wire-service headlines (Reuters/AP style), 6-12 words",
    "descriptive event summaries in past or present tense, 12-25 words, like a conflict event database entry",
    "local newspaper headlines naming towns, districts and officials",
    "headlines with attribution, e.g. ending '- police', '- minister' or 'says UN'",
    "country-prefixed headlines, e.g. 'Kenya: ...' or 'Sudan - ...'",
    "number-led headlines with casualty counts, money amounts or percentages",
    "tabloid or broadcast headlines, punchy and informal",
    "live-blog and breaking-news headlines, e.g. 'BREAKING:' or 'Live updates:'",
    "question or quote headlines",
    "headlines translated from non-English press, slightly unusual phrasing",
]

SYSTEM_PROMPT = f"""\
You write realistic news headlines used as NEGATIVE training examples for a classifier \
that detects water conflict in news.

{WATER_CONFLICT_DEFINITION}

Every headline you write must NOT be a water conflict under that definition. Specifically:
- Never combine water (or water infrastructure) with violence, protest, sabotage, threats, \
seizure or disputes involving force.
- A protest or clash must have no water cause; an attack must not damage or target water systems.
- Water-related headlines must describe peaceful events: policy, business, science, sport, \
weather, cooperation, engineering.

Write like real newsrooms, not like an AI: vary length, structure, tense and capitalisation; \
use specific place names, people, numbers and organisations; no repeated openings; \
no clichés like "amid tensions", "sparks outrage", "raises concerns". \
Never add reassurances that signal the label, such as "no clashes reported", "peacefully", \
"without incident" or "no violence"; real headlines do not say what did not happen. \
Invent plausible events; do not reuse the example headlines."""


def load_seed_titles() -> list[str]:
    titles = []
    for path in sorted(SERPAPI_RAW_DIR.glob("*.json")):
        for item in json.loads(path.read_text()).get("news_results", []):
            if item.get("title"):
                titles.append(item["title"])
    return titles


def build_spec(call_index: int, base_seed: int, seed_titles: list[str]) -> dict:
    """Deterministic diversity spec for one call."""
    rng = random.Random(f"{base_seed}-{call_index}")
    category = rng.choices([c for c, _ in CATEGORY_WEIGHTS], weights=[w for _, w in CATEGORY_WEIGHTS])[0]
    return {
        "call": call_index,
        "category": category,
        "subtopic": rng.choice(SUBTOPICS[category]),
        "regions": rng.sample(REGIONS, 4),
        "style": rng.choice(STYLES),
        "examples": rng.sample(seed_titles, min(SEEDS_PER_CALL, len(seed_titles))),
    }


def user_prompt(spec: dict) -> str:
    examples = "\n".join(f"- {t}" for t in spec["examples"]) or "- (none)"
    return f"""\
Write {HEADLINES_PER_CALL} distinct headlines.

Topic: {spec["subtopic"]}
Places (spread across these, and their towns and provinces): {", ".join(spec["regions"])}
Style: {spec["style"]}

Real headlines for tone and texture only (different topics; do not copy):
{examples}"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "headlines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headlines"],
    "additionalProperties": False,
}


def generate(client: anthropic.Anthropic, spec: dict) -> list[str]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt(spec)}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["headlines"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=int, required=True, help="Total number of generation calls in the plan")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the diversity specs")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    print("=" * 80)
    print("Generating Synthetic Negatives (Claude Haiku 5.5)")
    print("=" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seed_titles = load_seed_titles()
    print(f"\n  ✓ {len(seed_titles)} real seed titles from {SERPAPI_RAW_DIR}")

    specs = [build_spec(i, args.seed, seed_titles) for i in range(args.calls)]
    pending = [s for s in specs if not (RAW_DIR / f"{args.seed}_{s['call']:04d}.json").exists()]

    # ------------------------------------------------------------------------
    # 1. GENERATE
    # ------------------------------------------------------------------------
    print(f"\n[1/2] Generating {len(pending)} calls ({len(specs) - len(pending)} cached)...")
    client = anthropic.Anthropic(api_key=load_env_key("ANTHROPIC_API_KEY"), max_retries=8)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, client, s): s for s in pending}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                headlines = future.result()
            except (anthropic.APIError, RuntimeError, json.JSONDecodeError) as e:
                print(f"  ✗ call {spec['call']}: {e}")
                continue
            path = RAW_DIR / f"{args.seed}_{spec['call']:04d}.json"
            path.write_text(json.dumps({**spec, "model": MODEL, "headlines": headlines}))
            print(f"  ✓ call {spec['call']:3d} [{spec['category']}] {len(headlines)} headlines")

    # ------------------------------------------------------------------------
    # 2. COLLECT
    # ------------------------------------------------------------------------
    print("\n[2/2] Collecting...")
    seen = set()
    rows = []
    duplicates = 0
    for spec in specs:
        path = RAW_DIR / f"{args.seed}_{spec['call']:04d}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for headline in result["headlines"]:
            headline = " ".join(headline.split())
            key = normalize(headline)
            if not headline or key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append({
                "Headline": headline,
                "category": result["category"],
                "subtopic": result["subtopic"],
                "regions": "; ".join(result["regions"]),
                "style": result["style"],
                "call": result["call"],
            })

    with open(SYNTHETIC_HEADLINES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Headline", "category", "subtopic", "regions", "style", "call"])
        writer.writeheader()
        writer.writerows(rows)

    by_category = {}
    for r in rows:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
    print(f"  ✓ {len(rows):,} headlines {by_category} ({duplicates} duplicates dropped)")
    print(f"  ✓ Saved to: {SYNTHETIC_HEADLINES_PATH}")
    print("\n  Next: uv run verify_negatives.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
