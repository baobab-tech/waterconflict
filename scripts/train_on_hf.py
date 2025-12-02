#!/usr/bin/env python
# /// script
# dependencies = [
#     "water-conflict-classifier>=0.1.22",
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
    - Training data uploaded to HF Hub (see prepare_training_dataset.py)
    - HF authentication token

Versioning:
    - Model versions: v1.0, v1.1, v2.0, etc. (for trained models)
    - Dataset versions: d1.0, d1.1, d2.0, etc. (for training datasets)
    - Each model training automatically records which dataset version was used
    - Compare model performance across different model/dataset version combinations

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

Dataset-Model Version Tracking:
    The training script automatically detects and logs which dataset version (d1.0, d1.1, etc.)
    was used to train each model version (v1.0, v1.1, etc.). This mapping is recorded in:
    - experiment_history.jsonl (local)
    - Model card on HF Hub
    - Evals dataset on HF Hub
    
    Example: Model v2.3 trained on dataset d1.5 can be compared to model v2.4 trained on d2.0
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
from data_prep import load_training_data_hub, LABEL_NAMES
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
TRAINING_DATASET_REPO_NAME = os.environ.get("TRAINING_DATASET_REPO_NAME", "water-conflict-training-data")
MODEL_REPO_NAME = os.environ.get("MODEL_REPO_NAME", "water-conflict-classifier")
EVALS_REPO_NAME = os.environ.get("EVALS_REPO_NAME", "water-conflict-classifier-evals")

if not HF_ORGANIZATION:
    print("\n✗ Configuration not found!")
    print("  Set HF_ORGANIZATION environment variable\n")
    sys.exit(1)

TRAINING_DATASET_REPO = f"{HF_ORGANIZATION}/{TRAINING_DATASET_REPO_NAME}"
MODEL_REPO = f"{HF_ORGANIZATION}/{MODEL_REPO_NAME}"
EVALS_REPO = f"{HF_ORGANIZATION}/{EVALS_REPO_NAME}"

# Training configuration
BASE_MODEL = "BAAI/bge-small-en-v1.5" # "sentence-transformers/all-MiniLM-L6-v2" #
BATCH_SIZE = 16  # Smaller batches work better for ~1200 samples
NUM_EPOCHS = 3   # SetFit best practice: 3-5 epochs for embedding fine-tuning
NUM_ITERATIONS = 20
SAMPLING_STRATEGY = "undersampling" # "oversampling" or "undersampling"

# Versioning configuration
VERSION = os.environ.get("MODEL_VERSION")  # Optional: set explicit version (e.g., "v1.5")
VERSION_BUMP = os.environ.get("VERSION_BUMP", "minor")  # "major" or "minor" for auto-increment
AUTO_VERSION = VERSION is None  # Auto-increment if not specified
EXPERIMENT_HISTORY_FILE = "experiment_history.jsonl"

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
    print("\n[1/7] Authenticating with Hugging Face Hub...")
    if not setup_authentication():
        return
    
    # Step 1: Load training-ready data
    print(f"\n[2/7] Loading training data from HF Hub...")
    print(f"  Dataset: {TRAINING_DATASET_REPO}")
    print(f"  Note: Data is already preprocessed, balanced, and split")
    
    train_data, test_data = load_training_data_hub(TRAINING_DATASET_REPO)
    
    # Detect dataset version from HF Hub tags
    dataset_version = None
    try:
        api = HfApi()
        refs = api.list_repo_refs(repo_id=TRAINING_DATASET_REPO, repo_type="dataset")
        if hasattr(refs, 'tags') and refs.tags:
            # Get latest dataset version tag (format: d1.0, d1.1, etc.)
            dataset_tags = [tag.name for tag in refs.tags if tag.name.startswith('d')]
            if dataset_tags:
                # Parse and sort versions
                versions = []
                for tag in dataset_tags:
                    try:
                        parts = tag.lstrip('d').split('.')
                        major_num = int(parts[0])
                        minor_num = int(parts[1]) if len(parts) > 1 else 0
                        versions.append((major_num, minor_num, tag))
                    except (ValueError, IndexError):
                        continue
                if versions:
                    versions.sort()
                    dataset_version = versions[-1][2]  # Latest version
    except Exception as e:
        print(f"  ⚠ Could not detect dataset version: {e}")
    
    if dataset_version:
        print(f"  ✓ Using dataset version: {dataset_version}")
    else:
        print(f"  ⚠ No dataset version detected (dataset may not be versioned yet)")
    
    # Show label distribution
    label_counts = np.array(train_data['labels'].tolist()).sum(axis=0)
    print("\n  Label distribution in training set:")
    for name, count in zip(LABEL_NAMES, label_counts):
        print(f"    - {name}: {int(count)} ({count/len(train_data)*100:.1f}%)")
    
    neg_count = (train_data['labels'].apply(lambda x: x == [0, 0, 0])).sum()
    print(f"    - Negatives: {neg_count} ({neg_count/len(train_data)*100:.1f}%)")
    
    # Step 2: Determine version
    print(f"\n[3/7] Determining model version...")
    
    if AUTO_VERSION:
        is_major = VERSION_BUMP.lower() == "major"
        # Query HF Hub for existing tags
        try:
            api = HfApi()
            existing_tags = [ref.name for ref in api.list_repo_refs(MODEL_REPO, repo_type="model").tags]
            if existing_tags:
                # Parse version tags (format: v1.0, v1.1, etc.)
                versions = []
                for tag in existing_tags:
                    if tag.startswith('v'):
                        try:
                            parts = tag.lstrip('v').split('.')
                            major_num = int(parts[0])
                            minor_num = int(parts[1]) if len(parts) > 1 else 0
                            versions.append((major_num, minor_num, tag))
                        except (ValueError, IndexError):
                            continue
                
                if versions:
                    # Get latest version
                    versions.sort()
                    last_major, last_minor, _ = versions[-1]
                    
                    if is_major:
                        version = f"v{last_major + 1}.0"
                    else:
                        version = f"v{last_major}.{last_minor + 1}"
                    print(f"  ✓ Latest version on Hub: v{last_major}.{last_minor}")
                else:
                    version = "v1.0"
                    print(f"  ℹ No valid version tags found, starting at v1.0")
            else:
                version = "v1.0"
                print(f"  ℹ No tags found on Hub, starting at v1.0")
        except Exception as e:
            # Fallback to local history file
            print(f"  ⚠ Could not query Hub tags ({e}), falling back to local history")
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
    
    # Convert to HF Dataset format
    train_dataset = Dataset.from_pandas(train_data[['text', 'labels']])  # type: ignore
    test_dataset = Dataset.from_pandas(test_data[['text', 'labels']])  # type: ignore
    
    # Step 3: Train
    print(f"\n[4/7] Training model...")
    print(f"  (Estimated time: ~2-5 minutes on A10G GPU)\n")
    
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
        sampling_strategy=SAMPLING_STRATEGY, # "oversampling" or "undersampling"
        model_card_data=model_card_data
    )
    
    # Step 4: Evaluate
    print("\n[5/7] Evaluating model...")
    
    eval_results = evaluate_model(
        model=model,
        test_texts=test_data['text'].tolist(),
        test_labels=test_data['labels'].tolist(),
        label_names=LABEL_NAMES
    )
    
    print_evaluation_results(eval_results, LABEL_NAMES)
    
    # Step 5: Generate model card
    print("\n[6/7] Generating model card...")
    
    model_card = generate_model_card(
        model_repo=MODEL_REPO,
        base_model=BASE_MODEL,
        label_names=LABEL_NAMES,
        eval_results=eval_results,
        train_size=len(train_data),
        test_size=len(test_data),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        sampling_strategy=SAMPLING_STRATEGY, # "oversampling" or "undersampling"
        evals_repo=EVALS_REPO,
        training_dataset_repo=TRAINING_DATASET_REPO,
        dataset_version=dataset_version
    )
    
    print("  ✓ Model card generated")
    
    # Step 6: Push to Hub
    print("\n[7/7] Pushing model to Hugging Face Hub...")
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
        print("\n  Logging experiment & creating version tag...")
        print(f"  Using version: {version}")
        
        # Log experiment to local history
        tracker = ExperimentTracker(EXPERIMENT_HISTORY_FILE)
        
        # Calculate fields from loaded data
        train_size = len(train_data)
        test_size = len(test_data)
        full_train_size = train_size + test_size
        test_split = test_size / full_train_size if full_train_size > 0 else None
        
        experiment = tracker.log_experiment(
            version=version,
            config={
                "base_model": BASE_MODEL,
                "train_size": train_size,
                "test_size": test_size,
                "full_train_size": full_train_size,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "sample_size": train_size,
                "sampling_strategy": SAMPLING_STRATEGY, # "oversampling" or "undersampling"
                "test_split": test_split,
                "dataset_version": dataset_version,
                "training_type": "standard",
            },
            metrics=eval_results,
            metadata={
                "model_repo": MODEL_REPO,
                "training_dataset_repo": TRAINING_DATASET_REPO,
                "dataset_version": dataset_version,
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
        print(f"\n  Uploading evaluation results to HF evals dataset...")
        upload_eval_results(
            repo_id=EVALS_REPO,
            version=version,
            config=experiment['config'],
            metrics=eval_results,
            metadata={
                "model_repo": MODEL_REPO,
                "training_dataset_repo": TRAINING_DATASET_REPO,
                "dataset_version": dataset_version,
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
        "Military group attack workers at the Kajaki Dam in Afghanistan",
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
