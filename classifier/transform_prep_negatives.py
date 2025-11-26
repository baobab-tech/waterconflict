"""
Prepare negative examples for water conflict classifier training.

Extracts random samples from ACLED data 'notes' column to use as 
non-water-conflict examples (negatives) for training.

Input:
- ../data/ACLED RAW_2024-01-01-2025-05-19.csv

Output:
- negatives.csv (2500 random rows with 'Headline' column)
"""

import pandas as pd
import re
import os

print("=" * 80)
print("Preparing Negative Examples for Classifier Training")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading ACLED data...")

data_path = "../data/ACLED RAW_2024-01-01-2025-05-19.csv"

# Read only the 'notes' column to save memory
df = pd.read_csv(data_path, usecols=['notes'])  # type: ignore

print(f"  ✓ Loaded {len(df):,} rows from ACLED data")

# ============================================================================
# 2. FILTER AND SAMPLE RANDOM ROWS
# ============================================================================
print("\n[2/4] Filtering and sampling random rows...")

# Remove any null values
df_clean = df[df['notes'].notna()].copy()
print(f"  ✓ {len(df_clean):,} rows with valid notes")

# Exclude rows containing the word "water" (case-insensitive)
# These should not be in negative examples for water conflict classifier
df_no_water = df_clean[~df_clean['notes'].str.contains('water', case=False, na=False)].copy()
excluded_count = len(df_clean) - len(df_no_water)
print(f"  ✓ Excluded {excluded_count:,} rows containing 'water'")
print(f"  ✓ {len(df_no_water):,} rows available for sampling")

# Sample 2500 random rows
sample_size = 2500
if len(df_no_water) < sample_size:
    print(f"  ⚠ Warning: Only {len(df_no_water)} rows available, using all")
    sample_size = len(df_no_water)

negatives = df_no_water.sample(n=sample_size, random_state=42)
print(f"  ✓ Sampled {len(negatives):,} random rows (water-free)")

# ============================================================================
# 3. CLEAN HEADLINES
# ============================================================================
print("\n[3/4] Cleaning headlines...")

# Remove "On [date], " prefix from notes
def clean_headline(text):
    """Remove 'On [date], ' prefix from ACLED notes."""
    # Pattern: "On" + anything + ", " at the start
    cleaned = re.sub(r'^On\s+[^,]+,\s*', '', text)
    return cleaned

negatives['cleaned_notes'] = negatives['notes'].apply(clean_headline)
print(f"  ✓ Removed date prefixes from headlines")

# Show example of cleaning
sample_original = negatives['notes'].iloc[0]
sample_cleaned = negatives['cleaned_notes'].iloc[0]
print(f"\n  Example transformation:")
print(f"    Before: {sample_original[:80]}...")
print(f"    After:  {sample_cleaned[:80]}...")

# ============================================================================
# 4. SAVE TO CSV
# ============================================================================
print("\n[4/4] Saving to CSV...")

# Use cleaned notes as headlines
negatives_output = pd.DataFrame({
    'Headline': negatives['cleaned_notes'].values
})

output_path = "../data/negatives.csv"
negatives_output.to_csv(output_path, index=False)

print(f"  ✓ Saved to: {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\n  Output file: {output_path}")
print(f"  Total rows: {len(negatives_output):,}")
print(f"\n  Example headlines:")
for i, headline in enumerate(negatives_output['Headline'].head(3), 1):
    headline_preview = headline[:70] + "..." if len(headline) > 70 else headline
    print(f"    {i}. {headline_preview}")
print("\n  ✓ Ready to use with train_setfit_headline_classifier.py")
print("=" * 80)

