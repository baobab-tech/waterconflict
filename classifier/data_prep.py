"""
Data loading for water conflict classifier.

Provides functions for loading training-ready datasets from HF Hub or local files.
Datasets should already be preprocessed, balanced, and split into train/test sets.
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download


LABEL_NAMES = ['Trigger', 'Casualty', 'Weapon']


def load_training_data_hub(dataset_repo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training-ready data from HF Hub dataset repository.
    
    Expects a preprocessed dataset with train.csv and test.csv files,
    where each file has 'text' and 'labels' columns ready for training.
    
    Args:
        dataset_repo: HF dataset repository ID (e.g. "org/water-conflict-training-data")
        
    Returns:
        Tuple of (train_df, test_df) with 'text' and 'labels' columns
    """
    print(f"  Loading training data from: {dataset_repo}")
    
    # Download CSV files from HF Hub
    train_path = hf_hub_download(
        repo_id=dataset_repo,
        filename="train.csv",
        repo_type="dataset"
    )
    test_path = hf_hub_download(
        repo_id=dataset_repo,
        filename="test.csv",
        repo_type="dataset"
    )
    
    # Load CSVs
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Parse labels column (stored as string representation of list)
    import ast
    train_data['labels'] = train_data['labels'].apply(ast.literal_eval)
    test_data['labels'] = test_data['labels'].apply(ast.literal_eval)
    
    print(f"  ✓ Loaded {len(train_data)} training examples")
    print(f"  ✓ Loaded {len(test_data)} test examples")
    
    return train_data, test_data


def load_training_data_local(train_path: str = "../data/train.csv",
                             test_path: str = "../data/test.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training-ready data from local CSV files.
    
    Expects preprocessed files with 'text' and 'labels' columns ready for training.
    
    Args:
        train_path: Path to train.csv
        test_path: Path to test.csv
        
    Returns:
        Tuple of (train_df, test_df) with 'text' and 'labels' columns
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Parse labels column (stored as string representation of list)
    import ast
    train_data['labels'] = train_data['labels'].apply(ast.literal_eval)
    test_data['labels'] = test_data['labels'].apply(ast.literal_eval)
    
    print(f"  ✓ Loaded {len(train_data)} training examples")
    print(f"  ✓ Loaded {len(test_data)} test examples")
    
    return train_data, test_data


# ============================================================================
# LEGACY FUNCTIONS (for source data preprocessing - use prepare_training_dataset.py script instead)
# ============================================================================

def load_source_data_hub(dataset_repo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw source data from HF Hub dataset repository.
    
    LEGACY: Use load_training_data_hub() instead for training.
    This function loads raw positives/negatives that need preprocessing.
    
    Args:
        dataset_repo: HF dataset repository ID (e.g. "org/water-conflict-source-data")
        
    Returns:
        Tuple of (positives_df, negatives_df)
    """
    print(f"  Loading source data from: {dataset_repo}")
    
    # Download CSV files from HF Hub
    positives_path = hf_hub_download(
        repo_id=dataset_repo,
        filename="positives.csv",
        repo_type="dataset"
    )
    negatives_path = hf_hub_download(
        repo_id=dataset_repo,
        filename="negatives.csv",
        repo_type="dataset"
    )
    
    # Load CSVs
    positives = pd.read_csv(positives_path)
    negatives = pd.read_csv(negatives_path)
    
    # Drop priority_sample from positives if present (used only for schema consistency in dataset)
    if 'priority_sample' in positives.columns:
        positives = positives.drop(columns=['priority_sample'])
    
    # Check for hard negatives (priority_sample column)
    priority_count = 0
    if 'priority_sample' in negatives.columns:
        priority_count = negatives['priority_sample'].sum()
    
    print(f"  ✓ Loaded {len(positives)} positive examples")
    print(f"  ✓ Loaded {len(negatives)} negative examples")
    if priority_count > 0:
        print(f"    - Including {int(priority_count)} hard negatives (water-related peaceful news)")
    
    return positives, negatives


def preprocess_source_data(positives: pd.DataFrame, 
                           negatives: pd.DataFrame,
                           balance_negatives: bool = True) -> pd.DataFrame:
    """
    Preprocess raw source data into multi-label format for SetFit.
    
    LEGACY: Use prepare_training_dataset.py script instead.
    This function is kept for backward compatibility and local experimentation.
    
    Strategy for hard negatives (water-related peaceful news):
    - If negatives has priority_sample column, ALWAYS include all priority_sample=True rows
    - Then balance remaining negatives to match positives count
    - This ensures hard negatives are never diluted by random sampling
    
    Args:
        positives: DataFrame with 'Headline' and 'Basis' columns
        negatives: DataFrame with 'Headline', 'Basis', and optional 'priority_sample' columns
        balance_negatives: If True, sample regular negatives to match positive count (hard negatives always included)
        
    Returns:
        Combined and shuffled DataFrame with 'text' and 'labels' columns
    """
    import numpy as np
    
    # Prepare positives: multi-label format [Trigger, Casualty, Weapon]
    positives = positives.copy()
    positives['text'] = positives['Headline']
    positives['labels'] = positives.apply(
        lambda row: [
            1 if 'Trigger' in str(row['Basis']) else 0,
            1 if 'Casualty' in str(row['Basis']) else 0,
            1 if 'Weapon' in str(row['Basis']) else 0
        ], 
        axis=1
    )
    
    # Prepare negatives: all labels [0, 0, 0]
    negatives = negatives.copy()
    negatives['text'] = negatives['Headline']
    negatives['labels'] = [[0, 0, 0]] * len(negatives)
    
    n_positives = len(positives)
    
    # Handle hard negatives if priority_sample column exists
    if 'priority_sample' in negatives.columns:
        hard_negatives = negatives[negatives['priority_sample'] == True].copy()  # type: ignore
        regular_negatives = negatives[negatives['priority_sample'] == False].copy()  # type: ignore
        
        print(f"  ✓ Found {len(hard_negatives)} hard negatives (always included)")
        
        # Balance regular negatives to match positives, accounting for hard negatives
        if balance_negatives and len(regular_negatives) > n_positives - len(hard_negatives):
            target_regular = max(n_positives - len(hard_negatives), 0)
            if target_regular > 0 and len(regular_negatives) > target_regular:
                regular_negatives = regular_negatives.sample(n=target_regular, random_state=42)
                print(f"  ✓ Sampled {len(regular_negatives)} regular negatives")
        
        # Combine: hard negatives + sampled regular negatives
        negatives = pd.concat([hard_negatives, regular_negatives], ignore_index=True)  # type: ignore
        print(f"  ✓ Total negatives: {len(negatives)} ({len(hard_negatives)} hard + {len(regular_negatives)} regular)")
    else:
        # No priority_sample column - use original logic
        if balance_negatives and len(negatives) > n_positives:
            negatives = negatives.sample(n=n_positives, random_state=42)
            print(f"  ✓ Balanced negatives to {len(negatives)} (matching positives)")
    
    # Combine and shuffle
    data: pd.DataFrame = pd.concat([  # type: ignore
        positives[['text', 'labels']], 
        negatives[['text', 'labels']]
    ], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  ✓ Total examples: {len(data)}")
    print(f"  ✓ Positives: {n_positives} ({n_positives/len(data)*100:.1f}%)")
    print(f"  ✓ Negatives: {len(negatives)} ({len(negatives)/len(data)*100:.1f}%)")
    
    return data

