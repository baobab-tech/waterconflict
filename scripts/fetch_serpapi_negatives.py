"""
Fetch real news headlines from Google News via SerpApi.

Real titles serve three purposes:
- style examples for generate_synthetic_negatives.py
- a real share of the training negatives
- a held-out real-only test set, to catch the classifier learning synthetic style

Plan entries are either Google News topic sections per edition (no query, so
results are not shaped by keywords) or keyword searches pinned to a date window
(spreads events across 2024-2026 instead of this week's news). Each entry costs
one SerpApi search. An edition other than US costs one more search to read its
homepage, which yields that edition's topic tokens and its front-page headlines.
Raw responses are cached in ../data/serpapi_raw/, so a re-run never spends a
search twice.

Water-related titles are kept: verify_negatives.py decides which are safe
negatives (e.g. utility news) and which are water conflict.

Output:
- ../data/real_headlines.csv  (Headline, kind, entry, gl, source, date, link, water_terms)

Usage:
    python fetch_serpapi_negatives.py --dry-run     # print plan and search cost, no API calls
    python fetch_serpapi_negatives.py --limit 2     # fetch the first 2 uncached entries
    python fetch_serpapi_negatives.py --match "us World" --match "us airstrike"
    python fetch_serpapi_negatives.py               # run the full plan

Requirements:
    SERPAPI_KEY in ../.env (or the environment)
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from negatives_common import REAL_HEADLINES_PATH, SERPAPI_RAW_DIR, WATER_TERMS, load_env_key, normalize

SEARCH_URL = "https://serpapi.com/search.json"
ACCOUNT_URL = "https://serpapi.com/account.json"

# Searches to keep in reserve after a full run
QUOTA_RESERVE = 50

TOPIC_EDITIONS = ["us", "gb", "in", "ng"]
TOPICS = ["World", "Business", "Technology", "Entertainment", "Sports", "Science", "Health"]

# (query, gl, after, before)
DATED_QUERIES = [
    ("airstrike OR shelling", "us", "2024-02-01", "2024-04-01"),
    ("protest police clash", "gb", "2024-06-01", "2024-08-01"),
    ("gunmen attack", "ng", "2024-10-01", "2024-12-01"),
    ("election violence", "ke", "2025-01-01", "2025-03-01"),
    ("militants attack soldiers", "pk", "2025-04-01", "2025-06-01"),
    ("riot curfew", "in", "2025-07-01", "2025-09-01"),
    ("rebels seize town", "us", "2025-10-01", "2025-12-01"),
    ("water utility OR water supply", "za", "2024-03-01", "2024-06-01"),
    ("dam OR reservoir project", "in", "2025-02-01", "2025-05-01"),
    ("flood OR drought", "au", "2025-08-01", "2025-11-01"),
    ("kidnapped OR abducted", "ng", "2026-01-01", "2026-03-01"),
    ("coup OR mutiny", "us", "2026-03-01", "2026-05-01"),
    ("workers strike walkout", "za", "2026-05-01", "2026-07-01"),
    ("cartel OR gang shooting", "us", "2026-07-01", "2026-09-01"),
    ("election results", "in", "2024-04-01", "2024-07-01"),
    ("court sentenced OR jailed", "gb", "2024-08-01", "2024-10-01"),
    ("football OR cricket", "in", "2025-05-01", "2025-07-01"),
    ("inflation OR fuel prices", "ke", "2026-02-01", "2026-04-01"),
    ("hospital OR outbreak", "ph", "2024-11-01", "2025-01-01"),
    ("water treaty OR water agreement", "us", "2026-04-01", "2026-07-01"),
]

# The probe search, already cached; its menu_links hold the US topic tokens
PROBE = ("airstrike", "us")

MIN_TITLE_CHARS = 30
MAX_TITLE_CHARS = 250


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def query_cache(query: str, gl: str) -> Path:
    return SERPAPI_RAW_DIR / f"{gl}__{slug(query)}.json"


def home_cache(gl: str) -> Path:
    return SERPAPI_RAW_DIR / f"home__{gl}.json"


def topic_cache(gl: str, topic: str) -> Path:
    return SERPAPI_RAW_DIR / f"topic__{gl}__{slug(topic)}.json"


def http_get_json(url: str, params: dict) -> dict:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full_url, timeout=60) as resp:
        return json.load(resp)


def searches_left(api_key: str) -> int:
    """Account API is free and does not count against quota."""
    return int(http_get_json(ACCOUNT_URL, {"api_key": api_key})["total_searches_left"])


def search(api_key: str, **params) -> dict:
    data = http_get_json(SEARCH_URL, {"engine": "google_news", "hl": "en", "api_key": api_key, **params})
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def topic_tokens(gl: str) -> dict:
    """Topic title -> token for an edition, from a cached page's menu_links."""
    path = query_cache(*PROBE) if gl == PROBE[1] else home_cache(gl)
    if not path.exists():
        return {}
    menu = json.loads(path.read_text()).get("menu_links", [])
    return {m["title"]: m["topic_token"] for m in menu if m.get("topic_token")}


def plan_steps() -> list[tuple[str, str, Path, dict]]:
    """(kind, label, cache file, request params) for every planned search, in order."""
    steps = []
    for gl in TOPIC_EDITIONS:
        if gl != PROBE[1]:
            steps.append(("home", f"{gl} homepage", home_cache(gl), {"gl": gl}))
        for topic in TOPICS:
            steps.append(("topic", f"{gl} {topic}", topic_cache(gl, topic), {"gl": gl, "topic": topic}))
    for query, gl, after, before in DATED_QUERIES:
        q = f"{query} after:{after} before:{before}"
        steps.append(("query", f"{gl} {q}", query_cache(q, gl), {"gl": gl, "q": q}))
    return steps


def iter_articles(response: dict):
    """Yield article dicts from news_results, including clustered 'stories' and highlights."""
    for item in response.get("news_results", []):
        if item.get("title"):
            yield item
        for story in item.get("stories", []) or []:
            if story.get("title"):
                yield story
        if (item.get("highlight") or {}).get("title"):
            yield item["highlight"]


def clean_title(title: str, source_name: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    # Some titles carry a trailing " - Publisher" or " | Publisher"
    for sep in (" - ", " | ", " – "):
        if source_name and title.endswith(sep + source_name):
            title = title[: -len(sep + source_name)]
    return title


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and cache status, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N uncached entries")
    parser.add_argument("--match", action="append", default=[], help="Only fetch entries whose label contains this (repeatable)")
    args = parser.parse_args()

    SERPAPI_RAW_DIR.mkdir(parents=True, exist_ok=True)
    steps = plan_steps()
    uncached = [s for s in steps if not s[2].exists()]
    pending = [s for s in uncached if not args.match or any(m in s[1] for m in args.match)]
    pending = pending[:args.limit] if args.limit is not None else pending

    print("=" * 80)
    print("Fetching Real Headlines from Google News (SerpApi)")
    print("=" * 80)
    print(f"\n  Plan: {len(steps)} searches ({len(steps) - len(uncached)} cached)")
    print(f"  To fetch this run: {len(pending)} searches")

    if args.dry_run:
        for kind, label, _, _ in pending:
            print(f"    [{kind:5}] {label}")
        return

    api_key = load_env_key("SERPAPI_KEY")

    # ------------------------------------------------------------------------
    # 1. FETCH
    # ------------------------------------------------------------------------
    if pending:
        left = searches_left(api_key)
        print(f"\n[1/3] Fetching... ({left} searches left on account)")
        if left - len(pending) < QUOTA_RESERVE and len(pending) > 5:
            sys.exit(f"  ✗ Run would leave fewer than {QUOTA_RESERVE} searches; aborting")
        for i, (kind, label, path, params) in enumerate(pending, 1):
            params = dict(params)
            if kind == "topic":
                token = topic_tokens(params["gl"]).get(params.pop("topic"))
                if not token:
                    print(f"  ✗ {label}: no topic token (homepage not fetched?)")
                    continue
                params["topic_token"] = token
            try:
                data = search(api_key, **params)
            except Exception as e:
                print(f"  ✗ {label}: {e}")
                continue
            path.write_text(json.dumps(data))
            print(f"  ✓ [{i}/{len(pending)}] {label}: {sum(1 for _ in iter_articles(data))} articles")
            time.sleep(0.5)
        print(f"  ✓ {searches_left(api_key)} searches left on account")
    else:
        print("\n[1/3] All entries cached, no searches needed")

    # ------------------------------------------------------------------------
    # 2. EXTRACT TITLES
    # ------------------------------------------------------------------------
    print("\n[2/3] Extracting titles...")
    sources = [("query", f"{PROBE[1]} {PROBE[0]}", query_cache(*PROBE))]
    sources += [(kind, label, path) for kind, label, path, _ in steps]
    seen = set()
    rows = []
    stats = {"total": 0, "length": 0, "duplicate": 0}
    for kind, label, path in sources:
        if not path.exists():
            continue
        response = json.loads(path.read_text())
        gl = response.get("search_parameters", {}).get("gl", "")
        for art in iter_articles(response):
            stats["total"] += 1
            source = art.get("source")
            source_name = source.get("name", "") if isinstance(source, dict) else (source or "")
            title = clean_title(art["title"], source_name)
            if not (MIN_TITLE_CHARS <= len(title) <= MAX_TITLE_CHARS):
                stats["length"] += 1
                continue
            key = normalize(title)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({
                "Headline": title,
                "kind": kind,
                "entry": label,
                "gl": gl,
                "source": source_name,
                "date": art.get("iso_date") or art.get("date", ""),
                "link": art.get("link", ""),
                "water_terms": bool(WATER_TERMS.search(title)),
            })
    water = sum(1 for r in rows if r["water_terms"])
    print(f"  ✓ {stats['total']:,} articles seen; dropped {stats['length']:,} too short/long, "
          f"{stats['duplicate']:,} duplicates")
    print(f"  ✓ {len(rows):,} headlines kept ({water:,} with water terms)")

    # ------------------------------------------------------------------------
    # 3. SAVE
    # ------------------------------------------------------------------------
    print("\n[3/3] Saving...")
    with open(REAL_HEADLINES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Headline", "kind", "entry", "gl", "source", "date", "link", "water_terms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {REAL_HEADLINES_PATH}")
    print("\n  Next: uv run verify_negatives.py ../data/real_headlines.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
