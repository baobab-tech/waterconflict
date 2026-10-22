#!/usr/bin/env python
# /// script
# dependencies = ["sentence-transformers>=2.2", "scikit-learn>=1.3", "pandas>=2.0"]
# ///
"""
Build the training-ready negatives pool from verified real and synthetic headlines.

Steps:
1. Keep headlines verify_negatives.py labelled not_water_conflict
2. Embed with the classifier's base model and drop near-duplicates
   (cosine >= DUPLICATE_THRESHOLD; real headlines win ties)
3. Drop headlines too close to any positive (cosine >= POSITIVE_THRESHOLD),
   written to a review file instead
4. Hold out real and synthetic negatives that never enter training, used to
   check the model is not keying on synthetic style
5. Select the pool with even coverage: k-means clusters per group
   (water-related / other), picked round-robin across clusters

Inputs:
- ../data/real_headlines_verified.csv
- ../data/synthetic_headlines_verified.csv
- ../data/hard_negatives_verified.csv
- ../data/positives.csv

Output:
- ../data/negatives_updated.csv   (Headline, Basis, priority_sample, source) - for upload_datasets.py
- ../data/holdout_negatives.csv   (Headline, Basis, priority_sample, source)
- ../data/negatives_review.csv    near-positive headlines left out, for manual review

Usage:
    uv run build_negatives.py
"""

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from negatives_common import DATA_DIR, normalize

BASE_MODEL = "BAAI/bge-small-en-v1.5"
DUPLICATE_THRESHOLD = 0.85
POSITIVE_THRESHOLD = 0.85
WATER_SHARE = 0.35        # share of the pool that is water-related (hard negatives)
HOLDOUT_PER_SOURCE = 300  # held-out negatives per source (real, synthetic)
RANDOM_STATE = 42


def load_inputs() -> pd.DataFrame:
    frames = []
    for name, source in [("real_headlines", "real"), ("synthetic_headlines", "synthetic"),
                         ("hard_negatives", "synthetic")]:
        df = pd.read_csv(DATA_DIR / f"{name}_verified.csv")
        df["source"] = source
        frames.append(df[["Headline", "verdict", "water_related", "source"]])
    df = pd.concat(frames, ignore_index=True)
    df["water_related"] = df["water_related"].astype(str).str.lower() == "true"
    return df


def drop_near_duplicates(emb: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy pass in row order; returns indices of rows kept."""
    kept = []
    kept_emb = np.empty_like(emb)
    for i, vec in enumerate(emb):
        if kept and float((kept_emb[:len(kept)] @ vec).max()) >= threshold:
            continue
        kept_emb[len(kept)] = vec
        kept.append(i)
    return np.array(kept)


def select_diverse(emb: np.ndarray, n: int, n_clusters: int) -> np.ndarray:
    """Pick n rows round-robin across k-means clusters, in random order within each cluster."""
    if n >= len(emb):
        return np.arange(len(emb))
    labels = KMeans(n_clusters=min(n_clusters, len(emb)), random_state=RANDOM_STATE, n_init="auto").fit_predict(emb)
    rng = np.random.default_rng(RANDOM_STATE)
    queues = [list(rng.permutation(np.where(labels == c)[0])) for c in range(labels.max() + 1)]
    picked = []
    while len(picked) < n:
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    return np.array(picked)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Negatives in the pool (default: number of positives)")
    args = parser.parse_args()

    print("=" * 80)
    print("Building Negatives Pool")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. LOAD VERIFIED HEADLINES
    # ------------------------------------------------------------------------
    print("\n[1/5] Loading verified headlines...")
    df = load_inputs()
    print(f"  ✓ {len(df):,} candidates {df['source'].value_counts().to_dict()}")
    print(f"  ✓ Verdicts: {df['verdict'].value_counts().to_dict()}")
    df = df[df["verdict"] == "not_water_conflict"]
    # Real first, so real headlines win near-duplicate ties
    df = df.assign(key=df["Headline"].map(normalize)).drop_duplicates("key")
    df = pd.concat([df[df["source"] == "real"], df[df["source"] != "real"]], ignore_index=True)
    print(f"  ✓ {len(df):,} not_water_conflict after exact dedupe")

    positives = pd.read_csv(DATA_DIR / "positives.csv")
    pool_size = args.pool_size or len(positives)

    # ------------------------------------------------------------------------
    # 2. NEAR-DUPLICATES
    # ------------------------------------------------------------------------
    print(f"\n[2/5] Removing near-duplicates (cosine >= {DUPLICATE_THRESHOLD})...")
    model = SentenceTransformer(BASE_MODEL)
    emb = model.encode(df["Headline"].tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    kept = drop_near_duplicates(emb, DUPLICATE_THRESHOLD)
    print(f"  ✓ Dropped {len(df) - len(kept):,}; {len(kept):,} remain")
    df, emb = df.iloc[kept].reset_index(drop=True), emb[kept]

    # ------------------------------------------------------------------------
    # 3. TOO CLOSE TO POSITIVES
    # ------------------------------------------------------------------------
    print(f"\n[3/5] Checking similarity to positives (cosine >= {POSITIVE_THRESHOLD})...")
    pos_emb = model.encode(positives["Headline"].tolist(), batch_size=256, normalize_embeddings=True)
    nearest = (emb @ pos_emb.T)
    near_idx = nearest.argmax(axis=1)
    near_sim = nearest.max(axis=1)
    too_close = near_sim >= POSITIVE_THRESHOLD
    review = df[too_close].assign(
        nearest_positive=positives["Headline"].values[near_idx[too_close]],
        similarity=near_sim[too_close].round(3),
    )
    review[["Headline", "source", "similarity", "nearest_positive"]].to_csv(DATA_DIR / "negatives_review.csv", index=False)
    print(f"  ✓ {too_close.sum():,} left out for review -> negatives_review.csv")
    df, emb = df[~too_close].reset_index(drop=True), emb[~too_close]

    # ------------------------------------------------------------------------
    # 4. HOLDOUT
    # ------------------------------------------------------------------------
    print(f"\n[4/5] Holding out {HOLDOUT_PER_SOURCE} real + {HOLDOUT_PER_SOURCE} synthetic negatives...")
    holdout_idx = []
    for source in ["real", "synthetic"]:
        candidates = df[df["source"] == source]
        holdout_idx += candidates.sample(n=min(HOLDOUT_PER_SOURCE, len(candidates)), random_state=RANDOM_STATE).index.tolist()
    holdout = pd.DataFrame({
        "Headline": df.loc[holdout_idx, "Headline"].values,
        "Basis": "",
        "priority_sample": df.loc[holdout_idx, "water_related"].values,
        "source": df.loc[holdout_idx, "source"].values,
    })
    holdout.to_csv(DATA_DIR / "holdout_negatives.csv", index=False)
    print(f"  ✓ {len(holdout):,} held out ({int(holdout['priority_sample'].sum())} water-related)")
    rest = ~df.index.isin(holdout_idx)
    df, emb = df[rest].reset_index(drop=True), emb[rest]

    # ------------------------------------------------------------------------
    # 5. DIVERSE SELECTION
    # ------------------------------------------------------------------------
    print(f"\n[5/5] Selecting {pool_size:,} negatives ({WATER_SHARE:.0%} water-related)...")
    parts = []
    for water, share in [(True, WATER_SHARE), (False, 1 - WATER_SHARE)]:
        group = np.where(df["water_related"] == water)[0]
        n = int(round(pool_size * share))
        picked = group[select_diverse(emb[group], n, n_clusters=max(n // 20, 2))]
        parts.append(df.iloc[picked])
        print(f"  ✓ {'water-related' if water else 'other':13}: {len(picked):,} of {len(group):,} "
              f"{df.iloc[picked]['source'].value_counts().to_dict()}")
    pool = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)

    output = pd.DataFrame({
        "Headline": pool["Headline"].values,
        "Basis": "",
        "priority_sample": pool["water_related"].values,
        "source": pool["source"].values,
    })
    output.to_csv(DATA_DIR / "negatives_updated.csv", index=False)
    print(f"\n  ✓ {len(output):,} negatives -> {DATA_DIR / 'negatives_updated.csv'}")
    print("\n  Next: python upload_datasets.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
