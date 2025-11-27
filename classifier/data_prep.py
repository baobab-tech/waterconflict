"""
Data loading and preprocessing for water conflict classifier.

Provides functions for loading data from local files or HF Hub,
and preprocessing into the format required by SetFit.
"""

import pandas as pd
import numpy as np
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


def preprocess_data(positives: pd.DataFrame, 
                    negatives: pd.DataFrame,
                    balance_negatives: bool = True) -> pd.DataFrame:
    """
    Preprocess raw data into multi-label format for SetFit.
    
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


def stratified_sample_for_training(data: pd.DataFrame,
                                   n_samples: int,
                                   min_samples_per_label: int = 100,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Sample training data ensuring each label has sufficient representation.
    
    Stratified sampling that maintains label proportions while ensuring
    minority labels (especially Weapon) get enough samples for effective training.
    
    Args:
        data: DataFrame with 'text' and 'labels' columns
        n_samples: Target number of samples (will be approximate)
        min_samples_per_label: Minimum samples to include for each label
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame with good label representation
    """
    np.random.seed(random_state)
    
    # Separate by label presence
    indices_by_label = {}
    for i, label_name in enumerate(LABEL_NAMES):
        # Get indices where this label is present
        mask = data['labels'].apply(lambda x: x[i] == 1)
        indices_by_label[label_name] = data[mask].index.tolist()
    
    # Get negative samples (all zeros)
    negative_mask = data['labels'].apply(lambda x: x == [0, 0, 0])
    negative_indices = data[negative_mask].index.tolist()
    
    # Calculate proportional samples per label
    total_label_samples = sum(len(indices) for indices in indices_by_label.values())
    samples_per_label = {}
    
    for label_name, indices in indices_by_label.items():
        proportion = len(indices) / total_label_samples
        target = max(int(n_samples * 0.5 * proportion), min_samples_per_label)
        # Cap at available samples
        samples_per_label[label_name] = min(target, len(indices))
    
    # Sample from each label
    selected_indices = set()
    for label_name, target_count in samples_per_label.items():
        label_indices = indices_by_label[label_name]
        sampled = np.random.choice(label_indices, size=target_count, replace=False)
        selected_indices.update(sampled)
    
    # Fill remaining with negatives to reach n_samples
    remaining = n_samples - len(selected_indices)
    if remaining > 0 and len(negative_indices) > 0:
        neg_sample_size = min(remaining, len(negative_indices))
        neg_sampled = np.random.choice(negative_indices, size=neg_sample_size, replace=False)
        selected_indices.update(neg_sampled)
    
    # Create sampled dataframe
    sampled_data = data.loc[list(selected_indices)].copy()
    sampled_data = sampled_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Print label distribution
    label_counts = np.array(sampled_data['labels'].tolist()).sum(axis=0)
    print(f"\n  Stratified sample label distribution:")
    for label_name, count in zip(LABEL_NAMES, label_counts):
        print(f"    - {label_name}: {int(count)} ({count/len(sampled_data)*100:.1f}%)")
    neg_count = (sampled_data['labels'].apply(lambda x: x == [0, 0, 0])).sum()
    print(f"    - Negatives: {neg_count} ({neg_count/len(sampled_data)*100:.1f}%)")
    
    return sampled_data

