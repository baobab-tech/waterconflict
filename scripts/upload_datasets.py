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
import io

# Add project root to path and load configuration
project_root = Path(__file__).parent.parent  # scripts/ -> waterconflict/
sys.path.insert(0, str(project_root))

try:
    from config import HF_ORGANIZATION, SOURCE_DATASET_REPO_NAME
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
    
    # Full repo ID (source dataset - large, unsampled)
    repo_id = f"{HF_ORGANIZATION}/{SOURCE_DATASET_REPO_NAME}"
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
    
    # Check for negatives_updated.csv (training-ready version with hard negatives)
    # If it exists, use it. Otherwise fall back to negatives.csv
    negatives_file = "negatives_updated.csv" if (data_dir / "negatives_updated.csv").exists() else "negatives.csv"
    
    if negatives_file == "negatives_updated.csv":
        print(f"  📊 Using {negatives_file} (includes hard negatives & balanced sampling)")
    else:
        print(f"  ⚠️  Using {negatives_file} (run generate_hard_negatives.py to create optimized version)")
    
    files_to_upload = [
        ("positives.csv", "positives.csv"),
        (negatives_file, "negatives.csv"),  # Always upload as negatives.csv to HF
    ]
    
    for local_file, remote_file in files_to_upload:
        local_path = data_dir / local_file
        
        if not local_path.exists():
            print(f"  ✗ File not found: {local_path}")
            continue
        
        # Load and ensure consistent schema
        df = pd.read_csv(local_path)
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        
        # Ensure priority_sample column exists in both files for schema consistency
        if 'priority_sample' not in df.columns:
            df['priority_sample'] = False  # All positives are False (not applicable)
            print(f"  Added priority_sample column to {local_file}")
        
        # Show composition for negatives file
        if local_file.startswith("negatives"):
            priority_count = df['priority_sample'].sum()
            acled_count = len(df) - priority_count
            print(f"  Uploading {local_file} ({file_size_mb:.2f} MB)")
            print(f"    - Total: {len(df)} negatives")
            print(f"    - Hard negatives (water/peaceful): {int(priority_count)} ({priority_count/len(df)*100:.1f}%)")
            print(f"    - ACLED negatives (general conflict): {acled_count}")
        else:
            print(f"  Uploading {local_file} ({file_size_mb:.2f} MB, {len(df)} examples)")
        
        # Upload from DataFrame to ensure consistent schema
        try:
            # Save to temporary buffer with consistent schema
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            api.upload_file(
                path_or_fileobj=buffer.read(),
                path_in_repo=remote_file,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
            print(f"    ✓ Uploaded as {remote_file}")
        except Exception as e:
            print(f"    ✗ Error uploading {local_file}: {e}")
            sys.exit(1)
    
    # Create dataset configuration file
    print(f"\n[3/3] Creating dataset configuration file...")
    
    # First upload a dataset_infos.json to fix the type inference issue
    dataset_info = """{
  "default": {
    "features": {
      "Headline": {
        "dtype": "string",
        "_type": "Value"
      },
      "Basis": {
        "dtype": "string",
        "_type": "Value"
      },
      "priority_sample": {
        "dtype": "bool",
        "_type": "Value"
      }
    }
  }
}"""
    
    try:
        api.upload_file(
            path_or_fileobj=dataset_info.encode(),
            path_in_repo="dataset_infos.json",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print(f"  ✓ dataset_infos.json created")
    except Exception as e:
        print(f"  ✗ Error creating dataset_infos.json: {e}")
    
    # Create README for the dataset
    print(f"\n[4/4] Creating dataset README...")
    
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
- `negatives.csv`: Non-conflict news headlines (pre-balanced with hard negatives)

### Data Format

Both files have consistent schema:

| Column | Description |
|--------|-------------|
| Headline | News headline text |
| Basis | For positives: comma-separated labels (Trigger, Casualty, Weapon). For negatives: empty string |
| priority_sample | Boolean - For negatives: True for hard negatives (water-related peaceful news), False for ACLED. For positives: always False (not applicable) |

Note: `priority_sample` exists in both files for schema consistency but only has meaning for negatives.

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

## Hard Negatives & Dataset Balance

The negatives dataset is pre-balanced and training-ready, including:

1. **Hard Negatives (~120 examples, ~15-20% of negatives)**: Water-related peaceful news that teaches the model "water ≠ conflict". These prevent false positives where any water mention triggers conflict classification.
   - Water infrastructure projects (peaceful development)
   - Water research and technology breakthroughs  
   - Water conservation initiatives and conferences
   - Environmental water management topics

2. **ACLED Negatives (~600 examples)**: General conflict news without water mentions. Sampled from full ACLED dataset for efficient training.

The `priority_sample` column identifies hard negatives (True) vs regular negatives (False). This balanced composition eliminates the need for complex sampling logic during training - the dataset is ready to use as-is

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
    print(f"  SOURCE_DATASET_REPO = '{repo_id}'")
    print("\n⚠️  Note: After upload, HF may need a few minutes to process the dataset.")
    print("   If the viewer still shows errors, try clicking 'Refresh' on the dataset page.")
    print("\n")

if __name__ == "__main__":
    main()

