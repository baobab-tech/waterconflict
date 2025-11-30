#!/usr/bin/env python
"""
Prepare Training Dataset for Water Conflict Classifier

This script creates the training-ready dataset from the source data:
1. Loads raw SOURCE data from HF Hub (positives + negatives with hard negatives)
2. Preprocesses and balances the data
3. Splits into train/test sets
4. Samples to optimal size for SetFit (~1200 examples)
5. Uploads to TRAINING dataset repo on HF Hub
6. Creates version tag (d1.0, d1.1, etc.) for tracking

The resulting TRAINING dataset is ready-to-use - no further preprocessing needed.

Dataset Versioning:
    - Dataset versions use format: d1.0, d1.1, d2.0, etc.
    - Auto-increments by default (minor: d1.0 -> d1.1)
    - Use --major for major version bump (d1.5 -> d2.0)
    - Specify exact version with --version d2.0
    - Version tags tracked on HF Hub for reproducibility

Usage:
    # Auto-increment (minor bump)
    python prepare_training_dataset.py
    
    # Major version bump
    python prepare_training_dataset.py --major
    
    # Specify exact version
    python prepare_training_dataset.py --version d2.0
    
    # Custom sample size
    python prepare_training_dataset.py --sample-size 1500
    
    # Skip sampling (use all data)
    python prepare_training_dataset.py --no-sampling

Requirements:
    - HF authentication: hf auth login
    - Source data uploaded to HF Hub (see upload_datasets.py)
    - Config file with HF_ORGANIZATION set
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from huggingface_hub import HfApi, login, create_repo, create_tag
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import HF_ORGANIZATION, SOURCE_DATASET_REPO_NAME, TRAINING_DATASET_REPO_NAME
except ImportError:
    print("\n✗ config.py not found in project root!")
    print("  Copy config.sample.py to config.py and update with your org/username\n")
    sys.exit(1)

# Add classifier to path to import data prep functions
classifier_path = project_root / "classifier"
sys.path.insert(0, str(classifier_path))

from data_prep import load_source_data_hub, preprocess_source_data, LABEL_NAMES

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_DATASET_REPO = f"{HF_ORGANIZATION}/{SOURCE_DATASET_REPO_NAME}"
TRAINING_DATASET_REPO = f"{HF_ORGANIZATION}/{TRAINING_DATASET_REPO_NAME}"

# Training dataset configuration
DEFAULT_SAMPLE_SIZE = 1200  # Optimal for SetFit
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Dataset versioning configuration
DATASET_VERSION = os.environ.get("DATASET_VERSION")  # Optional: set explicit version (e.g., "d1.0")
VERSION_BUMP = os.environ.get("VERSION_BUMP", "minor")  # "major" or "minor" for auto-increment
AUTO_VERSION = DATASET_VERSION is None  # Auto-increment if not specified

# ============================================================================
# AUTHENTICATION
# ============================================================================

def setup_authentication():
    """Authenticate with HF Hub using cached token or environment variable."""
    from huggingface_hub.utils import get_token
    
    token = os.environ.get("HF_TOKEN")
    if token:
        print("  ✓ Using HF_TOKEN from environment")
        login(token=token)
    else:
        # Check if already authenticated via 'hf auth login'
        cached_token = get_token()
        if cached_token:
            print("  ✓ Using cached credentials from 'hf auth login'")
            # Already authenticated, no need to call login()
        else:
            print("  ✗ No authentication found")
            print("  Run: hf auth login")
            return False
    return True

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def get_next_dataset_version(repo_id: str, major: bool = False) -> str:
    """
    Get next dataset version from HF Hub tags.
    
    Args:
        repo_id: HF dataset repo ID
        major: If True, increment major version (dX.0), else minor (dX.Y)
        
    Returns:
        Next version string (e.g., "d1.0", "d1.1")
    """
    try:
        api = HfApi()
        existing_tags = [ref.name for ref in api.list_repo_refs(repo_id, repo_type="dataset").tags]
        
        if existing_tags:
            # Parse version tags (format: d1.0, d1.1, etc.)
            versions = []
            for tag in existing_tags:
                if tag.startswith('d'):
                    try:
                        parts = tag.lstrip('d').split('.')
                        major_num = int(parts[0])
                        minor_num = int(parts[1]) if len(parts) > 1 else 0
                        versions.append((major_num, minor_num, tag))
                    except (ValueError, IndexError):
                        continue
            
            if versions:
                versions.sort()
                last_major, last_minor, _ = versions[-1]
                
                if major:
                    return f"d{last_major + 1}.0"
                else:
                    return f"d{last_major}.{last_minor + 1}"
        
        return "d1.0"  # First version
        
    except Exception as e:
        print(f"  ⚠ Could not query Hub tags ({e}), starting at d1.0")
        return "d1.0"


def create_dataset_version_tag(repo_id: str, 
                               version: str, 
                               config: dict,
                               train_size: int,
                               test_size: int) -> bool:
    """
    Create a git tag on HF dataset repository.
    
    Args:
        repo_id: Dataset repo ID
        version: Version identifier (e.g., "d1.0")
        config: Dataset preparation config
        train_size: Number of training samples
        test_size: Number of test samples
        
    Returns:
        True if successful
    """
    try:
        api = HfApi()
        
        # Check if tag exists
        try:
            refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
            existing_tags = [tag.ref for tag in refs.tags] if hasattr(refs, 'tags') else []
            if f"refs/tags/{version}" in existing_tags or version in existing_tags:
                print(f"  ℹ Tag {version} already exists")
                return True
        except Exception:
            pass
        
        # Build tag message
        tag_message = f"Dataset {version} | train={train_size} | test={test_size}"
        if 'sample_size' in config:
            tag_message += f" | sampled={config['sample_size']}"
        
        # Create tag
        create_tag(
            repo_id=repo_id,
            tag=version,
            tag_message=tag_message,
            repo_type="dataset"
        )
        
        print(f"  ✓ Created dataset version tag: {version}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "409" in error_msg or "already exists" in error_msg.lower():
            print(f"  ℹ Tag {version} already exists")
            return True
        print(f"  ✗ Failed to create tag: {e}")
        return False


def upload_training_dataset(train_data: pd.DataFrame, 
                            test_data: pd.DataFrame,
                            source_dataset_repo: str,
                            training_dataset_repo: str,
                            config: dict,
                            version: str) -> str:
    """
    Upload the prepared training dataset to HF Hub.
    
    Args:
        train_data: Training split DataFrame with 'text' and 'labels' columns
        test_data: Test split DataFrame with 'text' and 'labels' columns
        source_dataset_repo: Source dataset repo (e.g. "org/water-conflict-source-data")
        training_dataset_repo: Training dataset repo (e.g. "org/water-conflict-training-data")
        config: Dict with preparation parameters
        version: Dataset version (e.g., "d1.0")
        
    Returns:
        Repository ID of uploaded dataset
    """
    print(f"  Uploading to: {training_dataset_repo}")
    
    # Create dataset repo (if doesn't exist)
    try:
        create_repo(
            repo_id=training_dataset_repo,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        print(f"  ✓ Repository created/verified")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Upload train and test CSVs
    api = HfApi()
    
    # Save locally first
    train_path = "/tmp/train.csv"
    test_path = "/tmp/test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    commit_message = f"Update training data (train: {len(train_data)}, test: {len(test_data)})"
    
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=training_dataset_repo,
        repo_type="dataset",
        commit_message=commit_message,
    )
    
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=training_dataset_repo,
        repo_type="dataset",
        commit_message=commit_message,
    )
    
    print(f"  ✓ Uploaded train.csv ({len(train_data)} examples)")
    print(f"  ✓ Uploaded test.csv ({len(test_data)} examples)")
    
    # Create dataset card
    label_counts = np.array(train_data['labels'].tolist()).sum(axis=0)
    label_dist = "\n".join([
        f"- **{name}**: {int(count)} ({count/len(train_data)*100:.1f}%)" 
        for name, count in zip(LABEL_NAMES, label_counts)
    ])
    
    neg_count = (train_data['labels'].apply(lambda x: x == [0, 0, 0])).sum()
    
    dataset_card = f"""---
license: cc-by-nc-4.0
license_link: https://acleddata.com/eula
license_name: cc-by-nc-4.0
task_categories:
- text-classification
language:
- en
tags:
- water-conflict
- setfit
- multi-label
- training-ready
- non-commercial
size_categories:
- 1K<n<10K
---

# Water Conflict Training Dataset (Training-Ready)

**Version**: {version}

## 🔬 Experimental Research

> This experimental research draws on Pacific Institute's [Water Conflict Chronology](https://www.worldwater.org/water-conflict/), which tracks water-related conflicts spanning over 4,500 years of human history. The work is conducted independently and is not affiliated with Pacific Institute.

This dataset is designed to assist researchers in training models to classify water-related conflict events at scale. The Pacific Institute maintains the world's most comprehensive open-source record of water-related conflicts, documenting over 2,700 events across 4,500 years of history. 

**⚠️ Non-Commercial Use Only:** This dataset includes data derived from ACLED, which restricts use to non-commercial purposes. Commercial use requires separate permission from ACLED. This is not a commercial product and is not intended for commercial use.

## 📋 Dataset Description

This dataset contains **preprocessed, balanced, and split training data** ready for training the water conflict classifier. No additional preprocessing is needed.

## Dataset Details

- **Source Dataset**: [{source_dataset_repo}](https://huggingface.co/datasets/{source_dataset_repo})
- **Train Samples**: {len(train_data)}
- **Test Samples**: {len(test_data)}
- **Test Split**: {config['test_split']*100:.0f}%
- **Labels**: {', '.join(LABEL_NAMES)}

## What's Different from Source Data?

This dataset is the **training-ready** version of the source data:

1. ✅ **Preprocessed**: Positives converted to multi-label format, negatives labeled as [0,0,0]
2. ✅ **Balanced**: Hard negatives (water-related peaceful news) always included, ACLED negatives balanced
3. ✅ **Sampled**: Reduced to optimal size for SetFit training (~{config.get('sample_size', 'all')} examples)
4. ✅ **Split**: Pre-split into train/test sets with stratification
5. ✅ **Ready to Use**: Load and train directly, no additional preprocessing

## Label Distribution (Training Set)

{label_dist}
- **Negatives (no conflict)**: {neg_count} ({neg_count/len(train_data)*100:.1f}%)

## Preparation Configuration

```python
{config}
```

## Usage

```python
from datasets import load_dataset

# Load training-ready dataset
dataset = load_dataset("{training_dataset_repo}")

train = dataset['train']  # Ready to train
test = dataset['test']    # Ready to evaluate

# Each example has:
# - 'text': headline text
# - 'labels': list of [trigger, casualty, weapon] (0 or 1 each)
```

## Training

This dataset is optimized for SetFit multi-label classification:

```python
from setfit import SetFitModel
from datasets import load_dataset

# Load data
dataset = load_dataset("{training_dataset_repo}")

# Train SetFit model
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", multi_target_strategy="one-vs-rest")
model.train(dataset['train'])

# Evaluate
predictions = model.predict(dataset['test']['text'])
```

## Data Pipeline

```
Source Data (raw positives + negatives)
  ↓
preprocess_source_data() - combine, balance, label
  ↓
iterative_train_test_split() - multi-label stratified 85/15 split
  ↓
sample (optional) - reduce to optimal size
  ↓
Training Dataset (this dataset) - ready to use!
```

## Labels

- **Trigger**: Water resource as conflict trigger/cause
- **Casualty**: Water infrastructure as casualty/target  
- **Weapon**: Water used as weapon/tool of conflict

Multiple labels can apply to one headline (multi-label classification).

## 📊 Data Sources

### Positive Examples (Water Conflict Headlines)
Pacific Institute (2025). *Water Conflict Chronology*. Pacific Institute, Oakland, CA.  
https://www.worldwater.org/water-conflict/

### Negative Examples (Non-Water Conflict Headlines)
Armed Conflict Location & Event Data Project (ACLED).  
https://acleddata.com/

**Note:** Training negatives include synthetic "hard negatives" - peaceful water-related news (e.g., "New desalination plant opens", "Water conservation conference") to prevent false positives on non-conflict water topics.

## 🌍 About This Project

This dataset is part of independent experimental research drawing on the Pacific Institute's Water Conflict Chronology. The Pacific Institute maintains the world's most comprehensive open-source record of water-related conflicts, documenting over 2,700 events across 4,500 years of history.

## 📜 License

This derived training dataset is made available under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

**IMPORTANT - Source Data Restrictions:**

This dataset is derived from:
1. **Pacific Institute's Water Conflict Chronology** (positives) - Open-source with attribution requirement
2. **ACLED data** (negatives) - Subject to [ACLED's Terms of Use](https://acleddata.com/eula)

**ACLED specifically requires:**
- Non-commercial use only (commercial use requires written permission from ACLED)
- Proper attribution to ACLED
- Compliance with their End User License Agreement

**You are free to:**
- **Share** — copy and redistribute the material for non-commercial purposes
- **Adapt** — remix, transform, and build upon the material for non-commercial purposes

**Under the following terms:**
- **Attribution** — You must credit Baobab Tech, Pacific Institute, and ACLED with appropriate citations
- **NonCommercial** — You may not use this material for commercial purposes (per ACLED's terms)
- **Source Compliance** — You must comply with the original licensing terms of Pacific Institute and ACLED data

For commercial use, you must obtain separate permission from ACLED. Contact: https://acleddata.com/

## 📝 Citation

If you use this dataset in your work, please cite:

```bibtex
@misc{{waterconflict2025,
  title={{Water Conflict Training Dataset}},
  author={{Independent Experimental Research Drawing on Pacific Institute Water Conflict Chronology}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/datasets/{training_dataset_repo}}}}},
  note={{Training data from Pacific Institute Water Conflict Chronology and ACLED}}
}}
```

Please also cite the Pacific Institute's Water Conflict Chronology:

```bibtex
@misc{{pacificinstitute2025,
  title={{Water Conflict Chronology}},
  author={{Pacific Institute}},
  year={{2025}},
  address={{Oakland, CA}},
  url={{https://www.worldwater.org/water-conflict/}},
  note={{Accessed: [access date]}}
}}
```

And ACLED for the negative examples:

```bibtex
@misc{{acled2025,
  title={{Armed Conflict Location & Event Data Project}},
  author={{ACLED}},
  year={{2025}},
  url={{https://acleddata.com/}},
  note={{Accessed: [access date]}}
}}
```
"""
    
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=training_dataset_repo,
        repo_type="dataset",
        commit_message=commit_message,
    )
    
    print(f"  ✓ README created")
    
    # Create version tag
    create_dataset_version_tag(
        repo_id=training_dataset_repo,
        version=version,
        config=config,
        train_size=len(train_data),
        test_size=len(test_data)
    )
    
    print(f"  ✓ Dataset available at: https://huggingface.co/datasets/{training_dataset_repo}")
    
    return training_dataset_repo

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training-ready dataset from source data")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                       help=f"Number of training samples (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                       help=f"Test set proportion 0-1 (default: {TEST_SIZE})")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                       help=f"Random seed for reproducibility (default: {RANDOM_STATE})")
    parser.add_argument("--no-sampling", action="store_true",
                       help="Use all data without sampling")
    parser.add_argument("--version", type=str, default=DATASET_VERSION,
                       help="Dataset version (e.g., d1.0) - auto-increments if not specified")
    parser.add_argument("--major", action="store_true",
                       help="Major version bump (d1.0 -> d2.0) instead of minor (d1.0 -> d1.1)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Water Conflict Training Dataset Preparation")
    print("=" * 80)
    
    # Step 0: Authentication
    print("\n[1/7] Authenticating with Hugging Face Hub...")
    if not setup_authentication():
        return
    
    # Step 0.5: Determine dataset version
    print("\n[2/7] Determining dataset version...")
    
    if args.version:
        dataset_version = args.version
        print(f"  Using specified version: {dataset_version}")
    else:
        is_major = args.major or VERSION_BUMP.lower() == "major"
        dataset_version = get_next_dataset_version(TRAINING_DATASET_REPO, major=is_major)
        bump_type = "major" if is_major else "minor"
        print(f"  Auto-generated version ({bump_type} bump): {dataset_version}")
    
    # Step 1: Load source data
    print(f"\n[3/7] Loading source data from HF Hub...")
    print(f"  Source: {SOURCE_DATASET_REPO}")
    positives, negatives = load_source_data_hub(SOURCE_DATASET_REPO)
    
    # Step 2: Preprocess
    print("\n[4/7] Preprocessing data...")
    print("  - Converting positives to multi-label format")
    print("  - Balancing negatives (hard negatives always included)")
    data = preprocess_source_data(positives, negatives, balance_negatives=True)
    
    # Label distribution
    label_counts = np.array(data['labels'].tolist()).sum(axis=0)
    print("\n  Label distribution in full dataset:")
    for name, count in zip(LABEL_NAMES, label_counts):
        print(f"    - {name}: {int(count)} ({count/len(positives)*100:.1f}% of positives)")
    
    # Step 3: Train/test split with iterative stratification for multi-label
    print(f"\n[5/7] Splitting dataset ({int((1-args.test_size)*100)}% train / {int(args.test_size*100)}% test)...")
    print("  Using iterative stratification to ensure balanced label distribution...")
    
    # Prepare data for iterative stratification
    X = np.array(range(len(data))).reshape(-1, 1)  # Indices as features
    y = np.array(data['labels'].tolist())  # Multi-label matrix
    
    # Iterative stratification ensures all labels are balanced in train/test
    train_idx, _, test_idx, _ = iterative_train_test_split(
        X, y, test_size=args.test_size
    )
    
    # Extract indices and create datasets
    train_indices = train_idx.flatten()
    test_indices = test_idx.flatten()
    
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    
    print(f"  ✓ Training pool: {len(train_data)} examples")
    print(f"  ✓ Test set: {len(test_data)} examples")
    
    # Show label distribution in both splits
    train_label_counts = np.array(train_data['labels'].tolist()).sum(axis=0)
    test_label_counts = np.array(test_data['labels'].tolist()).sum(axis=0)
    print(f"\n  Label distribution:")
    print(f"  {'Label':<12} {'Train':<15} {'Test':<15}")
    print(f"  {'-'*42}")
    for i, name in enumerate(LABEL_NAMES):
        train_pct = train_label_counts[i]/len(train_data)*100
        test_pct = test_label_counts[i]/len(test_data)*100
        print(f"  {name:<12} {int(train_label_counts[i]):>4} ({train_pct:>5.1f}%)   {int(test_label_counts[i]):>4} ({test_pct:>5.1f}%)")
    
    # Step 4: Optional sampling
    if not args.no_sampling and len(train_data) > args.sample_size:
        print(f"\n[6/7] Sampling training data with stratification for balanced labels...")
        print(f"  Target: {args.sample_size} samples")
        
        # Stratified sampling to ensure balanced representation of all labels (especially Weapon)
        # Strategy: Sample proportionally from each label, ensuring minority classes well-represented
        
        # Calculate how many examples to sample per label (can overlap since multi-label)
        samples_per_label = args.sample_size // 3  # Roughly equal per label
        
        sampled_indices = set()
        
        # Sample for each label
        for label_idx, label_name in enumerate(LABEL_NAMES):
            # Get examples with this label
            has_label = train_data[train_data['labels'].apply(lambda x: x[label_idx] == 1)]
            
            # Sample proportionally (more for minority classes like Weapon)
            n_to_sample = min(len(has_label), samples_per_label)
            label_sample = has_label.sample(n=n_to_sample, random_state=args.random_state + label_idx)
            sampled_indices.update(label_sample.index)
            
            print(f"  - {label_name}: sampled {n_to_sample} from {len(has_label)} available")
        
        # If we haven't reached target, add random negatives
        current_size = len(sampled_indices)
        if current_size < args.sample_size:
            remaining = args.sample_size - current_size
            negatives = train_data[train_data['labels'].apply(lambda x: x == [0, 0, 0])]
            available_negatives = negatives[~negatives.index.isin(sampled_indices)]
            
            if len(available_negatives) > 0:
                n_neg_sample = min(len(available_negatives), remaining)
                neg_sample = available_negatives.sample(n=n_neg_sample, random_state=args.random_state)
                sampled_indices.update(neg_sample.index)
                print(f"  - Negatives: sampled {n_neg_sample} from {len(available_negatives)} available")
        
        # Create final sampled dataset
        train_data = train_data.loc[list(sampled_indices)].reset_index(drop=True)
        
        # Show final label distribution
        final_label_counts = np.array(train_data['labels'].tolist()).sum(axis=0)
        print(f"\n  ✓ Sampled to {len(train_data)} examples (stratified by label)")
        print(f"  Label distribution in sampled data:")
        for name, count in zip(LABEL_NAMES, final_label_counts):
            print(f"    - {name}: {int(count)} ({count/len(train_data)*100:.1f}%)")
    else:
        print(f"\n[6/7] Using all training data (no sampling)...")
        print(f"  ✓ Training set: {len(train_data)} examples")
    
    # Step 5: Upload to HF Hub
    print(f"\n[7/7] Uploading training dataset to HF Hub...")
    print(f"  Version: {dataset_version}")
    
    config = {
        "dataset_version": dataset_version,
        "source_repo": SOURCE_DATASET_REPO,
        "sample_size": args.sample_size if not args.no_sampling else "all",
        "test_split": args.test_size,
        "random_state": args.random_state,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "preprocessing": "balanced (hard negatives always included)",
    }
    
    training_repo = upload_training_dataset(
        train_data=train_data,
        test_data=test_data,
        source_dataset_repo=SOURCE_DATASET_REPO,
        training_dataset_repo=TRAINING_DATASET_REPO,
        config=config,
        version=dataset_version
    )
    
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE! 🎉")
    print("=" * 80)
    print(f"\nTraining dataset: https://huggingface.co/datasets/{training_repo}")
    print(f"\nTo use in training:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{training_repo}')")
    print(f"  train_data = dataset['train']")
    print(f"  test_data = dataset['test']")
    print("\n")

if __name__ == "__main__":
    main()

