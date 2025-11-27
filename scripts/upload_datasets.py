#!/usr/bin/env python
"""
Upload Water Conflict Training Datasets to Hugging Face Hub

This script uploads the positives.csv and negatives.csv files to HF Hub
so they can be accessed by the training job running on HF infrastructure.

Usage:
    python upload_datasets.py

Requirements:
    pip install huggingface_hub pandas
    hf auth login  # Run this first to authenticate
"""

import pandas as pd
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

# Add project root to path and load configuration
project_root = Path(__file__).parent.parent  # scripts/ -> waterconflict/
sys.path.insert(0, str(project_root))

try:
    from config import HF_ORGANIZATION, DATASET_REPO_NAME
except ImportError:
    print("\n✗ config.py not found in project root!")
    print("  Copy config.sample.py to config.py and update with your org/username\n")
    sys.exit(1)

REPO_TYPE = "dataset"

def main():
    print("=" * 80)
    print("Uploading Water Conflict Training Data to Hugging Face Hub")
    print("=" * 80)
    
    # Initialize API
    api = HfApi()
    
    # Get username
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"\n✓ Authenticated as: {username}")
    except Exception as e:
        print("\n✗ Not authenticated! Please run: hf auth login")
        sys.exit(1)
    
    # Full repo ID
    repo_id = f"{HF_ORGANIZATION}/{DATASET_REPO_NAME}"
    print(f"  Target: {repo_id}")
    
    # Create repository
    print(f"\n[1/3] Creating dataset repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            exist_ok=True,
            private=False  # Set to True if you want a private dataset
        )
        print(f"  ✓ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"  ✗ Error creating repository: {e}")
        sys.exit(1)
    
    # Upload files
    print(f"\n[2/3] Uploading dataset files...")
    
    data_dir = Path(__file__).parent.parent / "data"  # scripts/ -> waterconflict/ -> data/
    files_to_upload = [
        ("positives.csv", "positives.csv"),
        ("negatives.csv", "negatives.csv"),
    ]
    
    for local_file, remote_file in files_to_upload:
        local_path = data_dir / local_file
        
        if not local_path.exists():
            print(f"  ✗ File not found: {local_path}")
            continue
        
        # Check file size
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_file} ({file_size_mb:.2f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_file,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
            print(f"    ✓ Uploaded {remote_file}")
        except Exception as e:
            print(f"    ✗ Error uploading {local_file}: {e}")
            sys.exit(1)
    
    # Create README for the dataset
    print(f"\n[3/3] Creating dataset README...")
    
    readme_content = f"""---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- water-conflict
- setfit
- multi-label
size_categories:
- 1K<n<10K
---

# Water Conflict Training Dataset

This dataset contains labeled examples for training a multi-label water conflict classifier.

## Dataset Structure

### Files

- `positives.csv`: Water conflict headlines with labels (Trigger, Casualty, Weapon)
- `negatives.csv`: Non-conflict news headlines (includes synthetic hard negatives)

### Data Format

Both files have the same structure:

| Column | Description |
|--------|-------------|
| Headline | News headline text |
| Basis | For positives: comma-separated labels (Trigger, Casualty, Weapon). For negatives: empty string |

### Example Rows

**Positive example:**
```
Headline,Basis
"Water reservoir attacked in region X",Casualty
```

**Negative example:**
```
Headline,Basis
"Political protest unrelated to water","
```

## Labels

- **Trigger**: Water resource as conflict trigger
- **Casualty**: Water infrastructure as casualty/target
- **Weapon**: Water as weapon/tool of conflict

## Hard Negatives

The negatives dataset includes synthetic "hard negatives" - peaceful water-related news that superficially resembles water conflicts but lacks violence. These are critical for preventing false positives where the model might classify any water-related news as a conflict.

Examples:
- Water infrastructure projects (peaceful development)
- Water research and technology breakthroughs
- Water conservation initiatives and conferences
- Environmental water management topics

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Citation

If you use this dataset, please cite the original ACLED data source.
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print(f"  ✓ README created")
    except Exception as e:
        print(f"  ✗ Error creating README: {e}")
    
    print("\n" + "=" * 80)
    print("UPLOAD COMPLETE! 🎉")
    print("=" * 80)
    print(f"\nDataset available at: https://huggingface.co/datasets/{repo_id}")
    print(f"\nTo use in training script:")
    print(f"  DATASET_REPO = '{repo_id}'")
    print("\n")

if __name__ == "__main__":
    main()

