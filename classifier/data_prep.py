"""
Data loading and preprocessing for water conflict classifier.

Provides functions for loading data from local files or HF Hub,
and preprocessing into the format required by SetFit.
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download


LABEL_NAMES = ['Trigger', 'Casualty', 'Weapon']


def load_local_data(positives_path: str = "../data/positives.csv", 
                    negatives_path: str = "../data/negatives.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data from local CSV files.
    
    Args:
        positives_path: Path to positives.csv
        negatives_path: Path to negatives.csv
        
    Returns:
        Tuple of (positives_df, negatives_df)
    """
    positives = pd.read_csv(positives_path)
    negatives = pd.read_csv(negatives_path)
    
    print(f"  ✓ Loaded {len(positives)} positive examples")
    print(f"  ✓ Loaded {len(negatives)} negative examples")
    
    return positives, negatives


def load_hub_data(dataset_repo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data from HF Hub dataset repository.
    
    Args:
        dataset_repo: HF dataset repository ID (e.g. "org/dataset-name")
        
    Returns:
        Tuple of (positives_df, negatives_df)
    """
    print(f"  Loading from dataset: {dataset_repo}")
    
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
    
    print(f"  ✓ Loaded {len(positives)} positive examples")
    print(f"  ✓ Loaded {len(negatives)} negative examples")
    
    return positives, negatives


def preprocess_data(positives: pd.DataFrame, 
                    negatives: pd.DataFrame,
                    balance_negatives: bool = True) -> pd.DataFrame:
    """
    Preprocess raw data into multi-label format for SetFit.
    
    Args:
        positives: DataFrame with 'Headline' and 'Basis' columns
        negatives: DataFrame with 'Headline' column
        balance_negatives: If True, sample negatives to match positive count
        
    Returns:
        Combined and shuffled DataFrame with 'text' and 'labels' columns
    """
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
    
    # Optional: Balance negatives to match positives count
    n_positives = len(positives)
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

