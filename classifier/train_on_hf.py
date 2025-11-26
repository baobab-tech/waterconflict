#!/usr/bin/env python
# /// script
# dependencies = [
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "datasets>=2.14.0",
#     "setfit>=1.0.0",
#     "sentence-transformers>=2.2.0",
#     "scikit-learn>=1.3.0",
#     "huggingface-hub>=0.19.0",
#     "torch>=2.0.0",
# ]
# ///

"""
SetFit Multi-Label Water Conflict Classifier - HF Jobs Training Script

This script is optimized for running on Hugging Face Jobs infrastructure.
It loads data from HF Hub and pushes the trained model back to HF Hub.

TRAINING OPTIMIZATION:
SetFit is designed for few-shot learning (8-100 examples per class) and reaches
peak performance quickly. With large datasets (>1000 samples), training time
explodes due to contrastive pair generation. This script samples the data by
default for efficient training with minimal performance loss.

Usage with HF Jobs:
    hf jobs uv run \\
      --flavor a10g-large \\
      --timeout 2h \\
      --secrets HF_TOKEN \\
      --env HF_ORGANIZATION=your-org-name \\
      train_on_hf.py

Local testing:
    uv run train_on_hf.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from setfit import SetFitModelCardData
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login
import warnings
warnings.filterwarnings('ignore')

# Add project root to path and load configuration
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to load from config.py (local development), fall back to env vars (HF Jobs)
try:
    from config import HF_ORGANIZATION, DATASET_REPO_NAME, MODEL_REPO_NAME
    print("  ✓ Loaded config from config.py")
except ImportError:
    HF_ORGANIZATION = os.environ.get("HF_ORGANIZATION")
    DATASET_REPO_NAME = os.environ.get("DATASET_REPO_NAME", "water-conflict-training-data")
    MODEL_REPO_NAME = os.environ.get("MODEL_REPO_NAME", "water-conflict-classifier")
    
    if not HF_ORGANIZATION:
        print("\n✗ Configuration not found!")
        print("  For local development: Copy config.sample.py to config.py")
        print("  For HF Jobs: Set HF_ORGANIZATION environment variable\n")
        print("  Example HF Jobs command:")
        print("    hf jobs uv run \\")
        print("      --flavor a10g-large \\")
        print("      --timeout 2h \\")
        print("      --secrets HF_TOKEN \\")
        print("      --env HF_ORGANIZATION=your-org-name \\")
        print("      train_on_hf.py\n")
        sys.exit(1)
    
    print(f"  ✓ Loaded config from environment (org: {HF_ORGANIZATION})")

# Import shared modules (same directory)
from data_prep import load_hub_data, preprocess_data, LABEL_NAMES
from training_logic import train_model
from evaluation import evaluate_model, print_evaluation_results
from model_card import generate_model_card

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset and model repos from config
DATASET_REPO = f"{HF_ORGANIZATION}/{DATASET_REPO_NAME}"
MODEL_REPO = f"{HF_ORGANIZATION}/{MODEL_REPO_NAME}"

# Training configuration
BASE_MODEL = "BAAI/bge-small-en-v1.5"  # 33.4M params, fast inference

# SetFit is designed for few-shot learning (8-64 samples per class)
# With large datasets, sampling down gives similar performance with much faster training
USE_SAMPLE_TRAINING = True  # Set to False to use all data (slower, minimal benefit)
SAMPLE_SIZE = 600  # ~100 examples per label combination (if USE_SAMPLE_TRAINING=True)

BATCH_SIZE = 32  # Increased from 16 (faster with sampled data)
NUM_EPOCHS = 1  # Reduced from 3 (SetFit reaches plateau quickly)
TEST_SIZE = 0.15

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
            print("  Please run: hf auth login")
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
    
    # Step 3: Train/test split (BEFORE sampling for more robust test metrics)
    print(f"\n[3/8] Splitting dataset ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
    print(f"  Full dataset: {len(data)} examples")
    
    full_train_raw, test_data_raw = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=data['labels'].apply(lambda x: tuple(x))
    )
    
    # Ensure DataFrames are properly typed
    full_train: pd.DataFrame = pd.DataFrame(full_train_raw).reset_index(drop=True)
    test_data: pd.DataFrame = pd.DataFrame(test_data_raw).reset_index(drop=True)
    
    print(f"  ✓ Full training pool: {len(full_train)} examples")
    print(f"  ✓ Test set (held-out): {len(test_data)} examples")
    
    # Step 4: Optional sampling from training pool (SetFit performs best with smaller datasets)
    if USE_SAMPLE_TRAINING and len(full_train) > SAMPLE_SIZE:
        print(f"\n[4/8] Sampling training data for efficient training...")
        print(f"  Training pool: {len(full_train)} examples")
        print(f"  Sampling to: {SAMPLE_SIZE} examples")
        print(f"  Rationale: SetFit reaches peak performance with ~8-100 examples per class")
        print(f"  Test set: {len(test_data)} examples (unchanged, more robust metrics)")
        
        train_data_raw = full_train.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        print(f"\n[4/8] Using all training data (no sampling)...")
        train_data_raw = full_train
    
    # Ensure final training data is properly typed
    train_data: pd.DataFrame = pd.DataFrame(train_data_raw).reset_index(drop=True)
    
    print(f"  ✓ Final training set: {len(train_data)} examples")
    print(f"  ✓ Final test set: {len(test_data)} examples")
    
    # Convert to HF Dataset format
    train_df: pd.DataFrame = train_data[['text', 'labels']]  # type: ignore
    test_df: pd.DataFrame = test_data[['text', 'labels']]  # type: ignore
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Step 5: Train
    print(f"\n[5/8] Training model...")
    print(f"  (Estimated time: ~2-5 minutes on A10G GPU with sampled data)\n")
    
    # Configure model card metadata
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
        sampling_strategy="undersampling",
        model_card_data=model_card_data
    )
    
    # Step 6: Evaluate
    print("\n[6/8] Evaluating model...")
    
    eval_results = evaluate_model(
        model=model,
        test_texts=test_data['text'].tolist(),
        test_labels=test_data['labels'].tolist(),
        label_names=LABEL_NAMES
    )
    
    print_evaluation_results(eval_results, LABEL_NAMES)
    
    # Step 7: Generate model card
    print("\n[7/8] Generating model card...")
    
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
        sampling_strategy="undersampling"
    )
    
    print("  ✓ Model card generated")
    
    # Step 8: Push to Hub
    print("\n[8/8] Pushing model to Hugging Face Hub...")
    print(f"  Target repository: {MODEL_REPO}")
    
    try:
        # Push model to Hub
        model.push_to_hub(
            MODEL_REPO,
            commit_message=f"Training complete - F1: {eval_results['overall']['f1_micro']:.4f}",
            private=False,  # Set to True for private model
        )
        
        # Upload model card
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        
        print(f"  ✓ Model pushed to: https://huggingface.co/{MODEL_REPO}")
        
    except Exception as e:
        print(f"  ✗ Error pushing to Hub: {e}")
        print("  Model trained successfully but not uploaded.")
        print("  You can manually upload using: model.push_to_hub()")
    
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
    print("Model size: ~130MB")
    print("Inference speed: ~5-10ms per headline on CPU")
    print("\n")

if __name__ == "__main__":
    main()

