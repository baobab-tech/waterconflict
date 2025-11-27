#!/usr/bin/env python
# /// script
# dependencies = [
#     "water-conflict-classifier>=0.1.14",
#     # For development/testing before PyPI publish, use:
#     # "water-conflict-classifier @ git+https://github.com/yourusername/waterconflict.git#subdirectory=classifier",
# ]
# ///

"""
SetFit Multi-Label Water Conflict Classifier - HF Jobs Training Script

Uses the published water-conflict-classifier package from PyPI.
Package source code is in ../classifier/

Prerequisites:
    - Package published to PyPI: https://pypi.org/project/water-conflict-classifier/
    - Training data uploaded to HF Hub (see upload_datasets.py)
    - HF authentication token

Usage with HF Jobs:
    # Auto-increment minor version (v1.0 -> v1.1)
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 2h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      --namespace your-org-name \\
      scripts/train_on_hf.py
    
    # Specify exact version
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 2h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      --env MODEL_VERSION=v2.0 \\
      --namespace your-org-name \\
      scripts/train_on_hf.py
    
    # Auto-increment major version (v1.5 -> v2.0)
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 2h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      --env VERSION_BUMP=major \\
      --namespace your-org-name \\
      scripts/train_on_hf.py

Local testing:
    # Auto-increment (default: minor)
    uv run scripts/train_on_hf.py
    
    # Specify version
    MODEL_VERSION=v1.5 uv run scripts/train_on_hf.py
    
    # Major version bump
    VERSION_BUMP=major uv run scripts/train_on_hf.py
    
    Note: Still requires HF Hub authentication and uploaded datasets.
"""

import os
import sys
import pandas as pd
import numpy as np
from datasets import Dataset
from setfit import SetFitModelCardData
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login
import warnings
warnings.filterwarnings('ignore')

# Import from package modules
from data_prep import load_hub_data, preprocess_data, stratified_sample_for_training, LABEL_NAMES
from training_logic import train_model
from evaluation import evaluate_model, print_evaluation_results
from model_card import generate_model_card
from versioning import ExperimentTracker, create_hf_version_tag, get_next_version
from evals_upload import upload_eval_results
from huggingface_hub import create_repo

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_ORGANIZATION = os.environ.get("HF_ORGANIZATION")
DATASET_REPO_NAME = os.environ.get("DATASET_REPO_NAME", "water-conflict-training-data")
MODEL_REPO_NAME = os.environ.get("MODEL_REPO_NAME", "water-conflict-classifier")
EVALS_REPO_NAME = os.environ.get("EVALS_REPO_NAME", "water-conflict-classifier-evals")

if not HF_ORGANIZATION:
    print("\n✗ Configuration not found!")
    print("  Set HF_ORGANIZATION environment variable\n")
    sys.exit(1)

DATASET_REPO = f"{HF_ORGANIZATION}/{DATASET_REPO_NAME}"
MODEL_REPO = f"{HF_ORGANIZATION}/{MODEL_REPO_NAME}"
EVALS_REPO = f"{HF_ORGANIZATION}/{EVALS_REPO_NAME}"

# Training configuration
BASE_MODEL = "BAAI/bge-small-en-v1.5" # "sentence-transformers/all-MiniLM-L6-v2" # 
USE_SAMPLE_TRAINING = True
SAMPLE_SIZE = 1200  # Increased for better label representation, especially Weapon
MIN_SAMPLES_PER_LABEL = 100  # Ensure each label gets sufficient training examples (especially Weapon ~292 available)
BATCH_SIZE = 64  # Increased for faster training with larger dataset
NUM_EPOCHS = 1  # SetFit best practice: <1 epoch with more data
NUM_ITERATIONS = 20  # Reduced from default 20 (SetFit generates contrastive pairs, less needed with more data)
TEST_SIZE = 0.15

# Versioning configuration
VERSION = os.environ.get("MODEL_VERSION")  # Optional: set explicit version (e.g., "v1.5")
VERSION_BUMP = os.environ.get("VERSION_BUMP", "minor")  # "major" or "minor" for auto-increment
AUTO_VERSION = VERSION is None  # Auto-increment if not specified
EXPERIMENT_HISTORY_FILE = "experiment_history.jsonl"

# ============================================================================
# DATASET UPLOAD
# ============================================================================

def upload_training_dataset(train_data: pd.DataFrame, 
                            test_data: pd.DataFrame,
                            version: str,
                            base_dataset_repo: str,
                            sampling_config: dict) -> str:
    """
    Upload the actual sampled training dataset to HF Hub.
    
    Creates a versioned dataset showing exactly what data was used for training.
    
    Args:
        train_data: Training split DataFrame
        test_data: Test split DataFrame
        version: Model version (e.g. "0.1.0")
        base_dataset_repo: Original dataset repo (e.g. "org/water-conflict-training-data")
        sampling_config: Dict with sampling parameters
        
    Returns:
        Repository ID of uploaded dataset
    """
    # Create versioned dataset repo name
    org = base_dataset_repo.split('/')[0]
    base_name = base_dataset_repo.split('/')[1]
    # Strip 'v' prefix if present to avoid double-v (version is already "v1.0" format)
    version_suffix = version.lstrip('v')
    versioned_repo = f"{org}/{base_name}-v{version_suffix}"
    
    print(f"  Uploading to: {versioned_repo}")
    
    # Create dataset repo
    try:
        create_repo(
            repo_id=versioned_repo,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
    except Exception as e:
        print(f"  Note: {e}")
    
    # Upload train and test CSVs
    api = HfApi()
    
    # Save locally first
    train_path = "/tmp/train_sampled.csv"
    test_path = "/tmp/test_sampled.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=versioned_repo,
        repo_type="dataset",
    )
    
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=versioned_repo,
        repo_type="dataset",
    )
    
    # Create dataset card
    dataset_card = f"""---
tags:
- water-conflict
- setfit
- training-data
license: cc-by-nc-4.0
---

# Water Conflict Training Dataset - {version}

This dataset contains the **actual sampled data** used to train model version `{version}`.

## Dataset Details

- **Parent Dataset**: [{base_dataset_repo}](https://huggingface.co/datasets/{base_dataset_repo})
- **Model Version**: {version}
- **Train Samples**: {len(train_data)}
- **Test Samples**: {len(test_data)}

## Sampling Configuration

```python
{sampling_config}
```

## Splits

- `train.csv`: Training samples (stratified sample from parent dataset)
- `test.csv`: Test samples (held-out evaluation set)

## Labels

Multi-label classification with labels: {LABEL_NAMES}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{versioned_repo}")
train = dataset['train']
test = dataset['test']
```

## Reproducibility

This dataset represents the exact data used to train the model, ensuring full reproducibility.
Parent dataset was sampled using stratified sampling to ensure balanced label representation.
"""
    
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=versioned_repo,
        repo_type="dataset",
    )
    
    print(f"  ✓ Dataset uploaded: https://huggingface.co/datasets/{versioned_repo}")
    return versioned_repo

# ============================================================================
# AUTHENTICATION
# ============================================================================

def setup_authentication():
    """Authenticate with HF Hub using token from environment or interactive login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        print("  ✓ Using HF_TOKEN from environment")
        login(token=token)
    else:
        print("  ⚠ No HF_TOKEN found, attempting interactive login...")
        try:
            login()
        except Exception as e:
            print(f"  ✗ Authentication failed: {e}")
            return False
    return True

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("SetFit Multi-Label Water Conflict Classifier - HF Jobs Training")
    print("=" * 80)
    
    # Step 0: Authentication
    print("\n[0/8] Authenticating with Hugging Face Hub...")
    if not setup_authentication():
        return
    
    # Step 1: Load data
    print("\n[1/8] Loading data from Hugging Face Hub...")
    positives, negatives = load_hub_data(DATASET_REPO)
    
    # Step 2: Preprocess
    print("\n[2/8] Preprocessing data...")
    data = preprocess_data(positives, negatives, balance_negatives=True)
    
    # Label distribution
    label_counts = np.array(data['labels'].tolist()).sum(axis=0)
    print("\n  Label distribution in positives:")
    for name, count in zip(LABEL_NAMES, label_counts):
        print(f"    - {name}: {int(count)} ({count/len(positives)*100:.1f}%)")
    
    # Step 3: Train/test split
    print(f"\n[3/8] Splitting dataset ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
    
    full_train_raw, test_data_raw = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=data['labels'].apply(lambda x: tuple(x))  # type: ignore
    )
    
    full_train = pd.DataFrame(full_train_raw).reset_index(drop=True)
    test_data = pd.DataFrame(test_data_raw).reset_index(drop=True)
    
    print(f"  ✓ Full training pool: {len(full_train)} examples")
    print(f"  ✓ Test set (held-out): {len(test_data)} examples")
    
    # Step 4: Optional sampling with stratification
    if USE_SAMPLE_TRAINING and len(full_train) > SAMPLE_SIZE:
        print(f"\n[4/8] Sampling training data with stratification...")
        print(f"  Target samples: {SAMPLE_SIZE}")
        print(f"  Ensuring minimum {MIN_SAMPLES_PER_LABEL} samples per label")
        train_data = stratified_sample_for_training(
            full_train,
            n_samples=SAMPLE_SIZE,
            min_samples_per_label=MIN_SAMPLES_PER_LABEL,
            random_state=42
        )
    else:
        print(f"\n[4/8] Using all training data (no sampling)...")
        train_data = pd.DataFrame(full_train).reset_index(drop=True)
    
    print(f"  ✓ Final training set: {len(train_data)} examples")
    print(f"  ✓ Final test set: {len(test_data)} examples")
    
    # Step 4.5: Determine version and upload training dataset
    print(f"\n[4.5/10] Determining version and uploading training dataset...")
    
    # Determine version first so we can version the dataset
    if AUTO_VERSION:
        is_major = VERSION_BUMP.lower() == "major"
        version = get_next_version(
            EXPERIMENT_HISTORY_FILE,
            major=is_major,
            minor=not is_major
        )
        bump_type = "major" if is_major else "minor"
        print(f"  Auto-generated version ({bump_type} bump): {version}")
    else:
        version = VERSION
        print(f"  Using specified version: {version}")
    
    # Upload the actual training dataset with version
    sampling_config = {
        "sample_size": SAMPLE_SIZE if USE_SAMPLE_TRAINING else None,
        "min_samples_per_label": MIN_SAMPLES_PER_LABEL,
        "stratified": True,
        "test_split": TEST_SIZE,
        "full_train_pool": len(full_train),
        "train_samples_used": len(train_data),
        "test_samples": len(test_data)
    }
    
    training_dataset_repo = upload_training_dataset(
        train_data=train_data,
        test_data=test_data,
        version=version,
        base_dataset_repo=DATASET_REPO,
        sampling_config=sampling_config
    )
    
    # Convert to HF Dataset format
    train_df = train_data[['text', 'labels']]
    test_df = test_data[['text', 'labels']]
    train_dataset = Dataset.from_pandas(train_df)  # type: ignore
    test_dataset = Dataset.from_pandas(test_df)  # type: ignore
    
    # Step 5: Train
    print(f"\n[5/10] Training model...")
    print(f"  (Estimated time: ~2-5 minutes on A10G GPU with sampled data)\n")
    
    model_card_data = SetFitModelCardData(
        language="en",
        license="cc-by-nc-4.0",
    )
    
    model = train_model(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        base_model=BASE_MODEL,
        label_names=LABEL_NAMES,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        sampling_strategy="undersampling",
        model_card_data=model_card_data
    )
    
    # Step 6: Evaluate
    print("\n[6/10] Evaluating model...")
    
    eval_results = evaluate_model(
        model=model,
        test_texts=test_data['text'].tolist(),
        test_labels=test_data['labels'].tolist(),
        label_names=LABEL_NAMES
    )
    
    print_evaluation_results(eval_results, LABEL_NAMES)
    
    # Step 7: Generate model card
    print("\n[7/10] Generating model card...")
    
    model_card = generate_model_card(
        model_repo=MODEL_REPO,
        base_model=BASE_MODEL,
        label_names=LABEL_NAMES,
        eval_results=eval_results,
        train_size=len(train_data),
        test_size=len(test_data),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        test_split=TEST_SIZE,
        full_train_size=len(full_train),
        num_iterations=NUM_ITERATIONS,
        sampling_strategy="undersampling",
        evals_repo=EVALS_REPO,
        training_dataset_repo=training_dataset_repo
    )
    
    print("  ✓ Model card generated")
    
    # Step 8: Push to Hub
    print("\n[8/10] Pushing model to Hugging Face Hub...")
    print(f"  Target repository: {MODEL_REPO}")
    
    try:
        model.push_to_hub(
            MODEL_REPO,
            commit_message=f"Training complete - F1: {eval_results['overall']['f1_micro']:.4f}",
            private=False,
        )
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        
        print(f"  ✓ Model pushed to: https://huggingface.co/{MODEL_REPO}")
        
        # Version tagging & experiment tracking
        print("\n[9/10] Logging experiment & creating version tag...")
        print(f"  Using version: {version}")
        
        # Log experiment to local history
        tracker = ExperimentTracker(EXPERIMENT_HISTORY_FILE)
        experiment = tracker.log_experiment(
            version=version,
            config={
                "base_model": BASE_MODEL,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "full_train_size": len(full_train),
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "sample_size": SAMPLE_SIZE if USE_SAMPLE_TRAINING else None,
                "sampling_strategy": "undersampling",
                "test_split": TEST_SIZE,
            },
            metrics=eval_results,
            metadata={
                "model_repo": MODEL_REPO,
                "dataset_repo": DATASET_REPO,
                "training_dataset_repo": training_dataset_repo,
            }
        )
        print(f"  ✓ Logged to {EXPERIMENT_HISTORY_FILE}")
        
        # Create HF Hub tag
        create_hf_version_tag(
            model_repo=MODEL_REPO,
            version=version,
            metrics=eval_results,
            config=experiment['config']
        )
        
        # Upload evaluation results to HF evals dataset
        print(f"\n[10/10] Uploading evaluation results to HF evals dataset...")
        upload_eval_results(
            repo_id=EVALS_REPO,
            version=version,
            config=experiment['config'],
            metrics=eval_results,
            metadata={
                "model_repo": MODEL_REPO,
                "dataset_repo": DATASET_REPO,
                "training_dataset_repo": training_dataset_repo,
            }
        )
        
    except Exception as e:
        print(f"  ✗ Error pushing to Hub: {e}")
        print("  Model trained successfully but not uploaded.")
    
    # Example predictions
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    test_examples = [
        "Taliban attack workers at the Kajaki Dam in Afghanistan",
        "New water treatment plant opens in California",
        "Violent protests erupt over dam construction in Sudan",
        "Scientists discover new water filtration method"
    ]
    
    predictions = model.predict(test_examples)
    
    print("\n")
    for text, pred in zip(test_examples, predictions):  # type: ignore
        labels_detected = [LABEL_NAMES[i] for i, val in enumerate(pred) if val == 1]
        label_str = ", ".join(labels_detected) if labels_detected else "None"
        print(f"  Text: {text}")
        print(f"  Labels: {label_str}\n")
    
    print("=" * 80)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print(f"\nYour model is available at: https://huggingface.co/{MODEL_REPO}")
    print(f"Base model: {BASE_MODEL}")
    print("Inference speed: ~5-10ms per headline on CPU")
    print("\n")

if __name__ == "__main__":
    main()
