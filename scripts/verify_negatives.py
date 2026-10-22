#!/usr/bin/env python
# /// script
# dependencies = ["anthropic>=2.1"]
# ///
"""
Check candidate negative headlines with Claude Haiku 5.5 before training.

Each headline gets a verdict against the water conflict definition:
- not_water_conflict: safe to use as a negative
- water_conflict:     would be a positive; must not be used as a negative
- unsure:             ambiguous; dropped from negatives (review by hand if useful)

It also records whether the headline is water-related, so water-related
negatives can be flagged as hard negatives (priority_sample).

Works on both real titles (real_headlines.csv) and generated ones
(synthetic_headlines.csv). Verdicts are cached in ../data/verify_raw/ per batch.

Output: <input>_verified.csv with added 'verdict' and 'water_related' columns.

Usage:
    uv run verify_negatives.py ../data/synthetic_headlines.csv
    uv run verify_negatives.py ../data/real_headlines.csv ../data/synthetic_headlines.csv
"""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from negatives_common import DATA_DIR, WATER_CONFLICT_DEFINITION, load_env_key

MODEL = "claude-haiku-5-5"
RAW_DIR = DATA_DIR / "verify_raw"
BATCH_SIZE = 50

SYSTEM_PROMPT = f"""\
You label news headlines for a water conflict classifier.

{WATER_CONFLICT_DEFINITION}

For each headline return:
- verdict: "water_conflict" if it meets the definition, "not_water_conflict" if it clearly \
does not, "unsure" if a reasonable annotator could go either way.
- water_related: true if the headline concerns water, water infrastructure, water bodies, \
rain/floods/drought, or uses a water word in any sense.

Judge only what the headline says. A protest, clash or attack with no water element is \
not_water_conflict. If a headline has both a water element (water, dam, river, well, canal, \
pipe, rain, flood, drought...) and any violence, protest, threat or confrontation, the verdict \
is water_conflict or unsure, never not_water_conflict."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["water_conflict", "not_water_conflict", "unsure"]},
                    "water_related": {"type": "boolean"},
                },
                "required": ["id", "verdict", "water_related"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def label_batch(client: anthropic.Anthropic, headlines: list[str]) -> dict[str, dict]:
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Label every headline by its number:\n\n{numbered}"}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    labels = {}
    for item in json.loads(text)["labels"]:
        if 0 <= item["id"] < len(headlines):
            labels[headlines[item["id"]]] = {"verdict": item["verdict"], "water_related": item["water_related"]}
    return labels


def batch_path(headlines: list[str]) -> Path:
    # Prompt is part of the key so prompt changes re-label
    digest = hashlib.sha256((SYSTEM_PROMPT + "\n".join(headlines)).encode()).hexdigest()[:16]
    return RAW_DIR / f"{digest}.json"


def verify_file(client: anthropic.Anthropic, input_path: Path, workers: int):
    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))
    headlines = [r["Headline"] for r in rows]
    batches = [headlines[i:i + BATCH_SIZE] for i in range(0, len(headlines), BATCH_SIZE)]
    pending = [b for b in batches if not batch_path(b).exists()]
    print(f"\n  {input_path.name}: {len(rows):,} headlines, {len(pending)} of {len(batches)} batches to label")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(label_batch, client, b): b for b in pending}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                labels = future.result()
            except (anthropic.APIError, RuntimeError, json.JSONDecodeError) as e:
                print(f"  ✗ batch failed: {e}")
                continue
            batch_path(batch).write_text(json.dumps(labels))

    labels = {}
    for batch in batches:
        if batch_path(batch).exists():
            labels.update(json.loads(batch_path(batch).read_text()))

    counts = {}
    for r in rows:
        label = labels.get(r["Headline"], {"verdict": "", "water_related": ""})
        r["verdict"] = label["verdict"]
        r["water_related"] = label["water_related"]
        counts[r["verdict"] or "unlabeled"] = counts.get(r["verdict"] or "unlabeled", 0) + 1

    output_path = input_path.with_name(f"{input_path.stem}_verified.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    water = sum(1 for r in rows if r["water_related"] is True)
    print(f"  ✓ {counts}, {water} water-related")
    print(f"  ✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV files with a 'Headline' column")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    print("=" * 80)
    print("Verifying Negatives (Claude Haiku 5.5)")
    print("=" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=load_env_key("ANTHROPIC_API_KEY"), max_retries=8)
    for input_path in args.inputs:
        verify_file(client, input_path, args.workers)
    print("=" * 80)


if __name__ == "__main__":
    main()
