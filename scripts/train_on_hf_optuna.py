#!/usr/bin/env python
# /// script
# dependencies = [
#     "water-conflict-classifier>=0.1.21",
#     "optuna>=3.0.0",
# ]
# ///

"""
SetFit Multi-Label Water Conflict Classifier - HF Jobs Training with Optuna

Alternative training script using Optuna hyperparameter search.
Based on: https://www.philschmid.de/setfit-outperforms-gpt3

This approach:
- Runs N trials with different hyperparameter combinations
- Searches across learning rates, epochs, batch sizes, iterations
- Optionally searches across multiple base models
- Automatically selects the best configuration

Prerequisites:
    - Package published to PyPI with optuna support
    - Training data uploaded to HF Hub
    - HF authentication token

Usage with HF Jobs:
    # Quick search (20 trials, ~30 min)
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 2h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      --env N_TRIALS=20 \\
      --namespace your-org-name \\
      scripts/train_on_hf_optuna.py

    # Full search (50 trials, ~1-2 hours)
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 4h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      --env N_TRIALS=50 \\
      --namespace your-org-name \\
      scripts/train_on_hf_optuna.py

    # Comprehensive search with model selection (100 trials)
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 6h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      --env N_TRIALS=100 \\
      --env SEARCH_MODELS=true \\
      --namespace your-org-name \\
      scripts/train_on_hf_optuna.py

Environment Variables:
    HF_ORGANIZATION: Your HuggingFace organization (required)
    N_TRIALS: Number of Optuna trials (default: 50)
    SEARCH_MODELS: Whether to search across base models (default: false)
    MODEL_VERSION: Explicit version (default: auto-increment)
    VERSION_BUMP: 'major' or 'minor' for auto-increment (default: minor)
"""

import os
import sys
import numpy as np
from datasets import Dataset
from setfit import SetFitModelCardData
from huggingface_hub import HfApi, login
import warnings
warnings.filterwarnings('ignore')

# Import from package modules
from data_prep import load_training_data_hub, LABEL_NAMES
from training_logic_optuna import train_model_with_optuna
from evaluation import evaluate_model, print_evaluation_results
from model_card import generate_model_card
from versioning import ExperimentTracker, create_hf_version_tag, get_next_version
from evals_upload import upload_eval_results

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

# Optuna configuration
N_TRIALS = int(os.environ.get("N_TRIALS", "50"))
SEARCH_SAMPLE_SIZE = int(os.environ.get("SEARCH_SAMPLE_SIZE", "200"))  # Small sample for fast search
SEARCH_MODELS = os.environ.get("SEARCH_MODELS", "false").lower() == "true"
TIMEOUT = int(os.environ.get("TIMEOUT_SECONDS", "0")) or None  # 0 = no timeout

# Base models to search across (if SEARCH_MODELS=true)
BASE_MODELS = [
    "BAAI/bge-small-en-v1.5",       # 33M params, fast
    "BAAI/bge-base-en-v1.5",        # 110M params, better quality
    "sentence-transformers/all-mpnet-base-v2",  # 110M params, battle-tested
]

# Versioning configuration
VERSION = os.environ.get("MODEL_VERSION")
VERSION_BUMP = os.environ.get("VERSION_BUMP", "minor")
AUTO_VERSION = VERSION is None
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
    print("SetFit Water Conflict Classifier - Optuna Hyperparameter Search")
    print("=" * 80)

    # Step 0: Authentication
    print("\n[1/7] Authenticating with Hugging Face Hub...")
    if not setup_authentication():
        return

    # Step 1: Load training-ready data
    print(f"\n[2/7] Loading training data from HF Hub...")
    print(f"  Dataset: {TRAINING_DATASET_REPO}")

    train_data, test_data = load_training_data_hub(TRAINING_DATASET_REPO)

    # Detect dataset version from HF Hub tags
    dataset_version = None
    try:
        api = HfApi()
        refs = api.list_repo_refs(repo_id=TRAINING_DATASET_REPO, repo_type="dataset")
        if hasattr(refs, 'tags') and refs.tags:
            dataset_tags = [tag.name for tag in refs.tags if tag.name.startswith('d')]
            if dataset_tags:
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
                    dataset_version = versions[-1][2]
    except Exception as e:
        print(f"  ⚠ Could not detect dataset version: {e}")

    if dataset_version:
        print(f"  ✓ Using dataset version: {dataset_version}")

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
        try:
            api = HfApi()
            existing_tags = [ref.name for ref in api.list_repo_refs(MODEL_REPO, repo_type="model").tags]
            if existing_tags:
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
                    versions.sort()
                    last_major, last_minor, _ = versions[-1]

                    if is_major:
                        version = f"v{last_major + 1}.0"
                    else:
                        version = f"v{last_major}.{last_minor + 1}"
                    print(f"  ✓ Latest version on Hub: v{last_major}.{last_minor}")
                else:
                    version = "v1.0"
            else:
                version = "v1.0"
        except Exception as e:
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
    train_dataset = Dataset.from_pandas(train_data[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'labels']])

    # Step 3: Hyperparameter Search and Training
    print(f"\n[4/7] Running Optuna hyperparameter search...")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Search sample size: {SEARCH_SAMPLE_SIZE} (from {len(train_data)} total)")
    print(f"  Search models: {SEARCH_MODELS}")
    if TIMEOUT:
        print(f"  Timeout: {TIMEOUT}s")
    print(f"  Strategy: Search on small sample, train final on full data\n")

    model_card_data = SetFitModelCardData(
        language="en",
        license="cc-by-nc-4.0",
    )

    model, best_params = train_model_with_optuna(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        label_names=LABEL_NAMES,
        n_trials=N_TRIALS,
        search_sample_size=SEARCH_SAMPLE_SIZE,
        direction="maximize",
        metric="f1",
        base_models=BASE_MODELS,
        search_models=SEARCH_MODELS,
        model_card_data=model_card_data,
        timeout=TIMEOUT,
    )

    # Extract best base model (if model search was enabled)
    best_base_model = best_params.get("model_id", BASE_MODELS[0])

    # Step 4: Evaluate
    print("\n[5/7] Evaluating final model...")

    eval_results = evaluate_model(
        model=model,
        test_texts=test_data['text'].tolist(),
        test_labels=test_data['labels'].tolist(),
        label_names=LABEL_NAMES
    )

    print_evaluation_results(eval_results, LABEL_NAMES)

    # Step 5: Generate model card
    print("\n[6/7] Generating model card...")

    # Build training config from best params
    batch_size = best_params.get("batch_size", 16)
    num_epochs = best_params.get("num_epochs", 3)
    num_iterations = best_params.get("num_iterations", 20)
    sampling_strategy = best_params.get("sampling_strategy", "oversampling")

    model_card = generate_model_card(
        model_repo=MODEL_REPO,
        base_model=best_base_model,
        label_names=LABEL_NAMES,
        eval_results=eval_results,
        train_size=len(train_data),
        test_size=len(test_data),
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        sampling_strategy=sampling_strategy,
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
            commit_message=f"Optuna search ({N_TRIALS} trials) - F1: {eval_results['overall']['f1_micro']:.4f}",
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

        train_size = len(train_data)
        test_size = len(test_data)
        full_train_size = train_size + test_size
        test_split = test_size / full_train_size if full_train_size > 0 else None

        experiment = tracker.log_experiment(
            version=version,
            config={
                "base_model": best_base_model,
                "train_size": train_size,
                "test_size": test_size,
                "full_train_size": full_train_size,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "num_iterations": num_iterations,
                "sampling_strategy": sampling_strategy,
                "test_split": test_split,
                "dataset_version": dataset_version,
                "training_type": "optuna",
                "n_trials": N_TRIALS,
                "search_sample_size": SEARCH_SAMPLE_SIZE,
                "search_models": SEARCH_MODELS,
                "best_hyperparameters": best_params,
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
    for text, pred in zip(test_examples, predictions):
        labels_detected = [LABEL_NAMES[i] for i, val in enumerate(pred) if val == 1]
        label_str = ", ".join(labels_detected) if labels_detected else "None"
        print(f"  Text: {text}")
        print(f"  Labels: {label_str}\n")

    print("=" * 80)
    print("OPTUNA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nModel available at: https://huggingface.co/{MODEL_REPO}")
    print(f"Base model: {best_base_model}")
    print("\n")

if __name__ == "__main__":
    main()
