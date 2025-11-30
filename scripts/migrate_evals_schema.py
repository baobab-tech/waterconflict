#!/usr/bin/env python
"""
Throwaway migration script to update evals dataset schema.

Adds missing 'dataset_version' column to track which training data version (d1.0, d1.1, etc.)
was used for each model training run.

Usage:
    python scripts/migrate_evals_schema.py
"""

import os
import sys
import pandas as pd
from huggingface_hub import HfApi
from pathlib import Path

# Configuration - hardcoded for one-time use
EVALS_REPO = "baobabtech/water-conflict-classifier-evals"

def main():
    print("=" * 80)
    print("Evals Dataset Schema Migration")
    print("=" * 80)
    print(f"\nTarget: {EVALS_REPO}")
    print("\nThis will add 'dataset_version' column to track training data versions")
    
    # Initialize API and check authentication
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"\n✓ Authenticated as: {username}")
    except Exception as e:
        print("\n✗ Not authenticated! Please run: huggingface-cli login")
        sys.exit(1)
    
    # Download existing CSV
    print("\n[1/3] Downloading existing evals data...")
    try:
        from datasets import load_dataset
        ds = load_dataset(EVALS_REPO, split="train")
        df = ds.to_pandas()  # type: ignore
        print(f"  ✓ Loaded {len(df)} existing records")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return
    
    # Show current schema
    print(f"\n  Current columns: {list(df.columns)}")
    
    # Add dataset_version column if missing
    print("\n[2/3] Adding dataset_version column...")
    
    if 'dataset_version' not in df.columns:
        # Initialize with None - users can manually update historical records if needed
        df['dataset_version'] = None
        print("  ✓ Added 'dataset_version' column (initialized to None)")
        print("  ℹ Historical records will show None until manually updated")
    else:
        print("  ℹ Column already exists")
    
    # Reorder columns to put dataset_version after dataset_repo for readability
    desired_column_order = [
        'version', 'timestamp',
        'base_model', 'train_size', 'test_size', 'full_train_size',
        'batch_size', 'num_epochs', 'sample_size', 'sampling_strategy',
        'test_split', 'num_iterations',
        'f1_micro', 'f1_macro', 'f1_samples', 'accuracy', 'hamming_loss',
        'trigger_precision', 'trigger_recall', 'trigger_f1', 'trigger_support',
        'casualty_precision', 'casualty_recall', 'casualty_f1', 'casualty_support',
        'weapon_precision', 'weapon_recall', 'weapon_f1', 'weapon_support',
        'model_repo', 'dataset_repo', 'dataset_version', 'notes'
    ]
    
    # Only reorder columns that exist
    existing_ordered = [col for col in desired_column_order if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_ordered]
    df = df[existing_ordered + remaining]
    
    print(f"\n  New columns: {list(df.columns)}")
    
    # Upload updated CSV
    print("\n[3/3] Uploading updated schema...")
    
    csv_path = Path("evals_results_migrated.csv")
    df.to_csv(csv_path, index=False)
    
    try:
        api.upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo="evals_results.csv",
            repo_id=EVALS_REPO,
            repo_type="dataset",
            commit_message="Schema migration: Add dataset_version column"
        )
        print(f"  ✓ Uploaded to: https://huggingface.co/datasets/{EVALS_REPO}")
        
        # Clean up
        if csv_path.exists():
            csv_path.unlink()
        
        print("\n" + "=" * 80)
        print("MIGRATION COMPLETE! ✓")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Update classifier/evals_upload.py to include dataset_version")
        print("2. Future training runs will automatically capture dataset_version")
        print("3. Historical records show None (can be manually updated if needed)\n")
        
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        if csv_path.exists():
            csv_path.unlink()

if __name__ == "__main__":
    main()

