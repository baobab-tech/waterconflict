"""
Upload evaluation results to HuggingFace Hub for experiment tracking.

Provides utilities for:
- Uploading evaluation metrics to HF dataset
- Comparing experiments across different hyperparameters
- Tracking training configurations and results
"""

import pandas as pd
from datetime import datetime
from typing import Optional
from huggingface_hub import HfApi, create_repo
from pathlib import Path


def upload_eval_results(
    repo_id: str,
    version: str,
    config: dict,
    metrics: dict,
    metadata: Optional[dict] = None,
    token: Optional[str] = None
) -> bool:
    """
    Upload evaluation results to HuggingFace dataset for experiment tracking.
    
    Args:
        repo_id: HF dataset repo ID (e.g., "org/water-conflict-classifier-evals")
        version: Version identifier (e.g., "v1.0", "v1.1")
        config: Training configuration dict with keys like:
            - base_model: str
            - train_size: int
            - test_size: int
            - batch_size: int
            - num_epochs: int
            - sample_size: Optional[int]
            - sampling_strategy: str
            - test_split: float
            - num_iterations: int
        metrics: Evaluation metrics dict with structure:
            - overall: dict with f1_micro, f1_macro, accuracy, etc.
            - per_label: dict with per-label precision, recall, f1
            - y_true, y_pred: numpy arrays (will be ignored)
        metadata: Optional additional metadata dict with keys:
            - model_repo: Model repository ID
            - dataset_repo: Training dataset repository ID
            - dataset_version: Dataset version used (e.g., d1.0, d1.1)
            - notes: Optional notes
        token: Optional HF token (uses default if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import numpy as np
        api = HfApi(token=token)
        
        # Ensure repository exists
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
        except Exception as e:
            # Repo might already exist, continue
            pass
        
        # Load existing results if they exist
        existing_df = None
        try:
            from datasets import load_dataset
            existing_ds = load_dataset(repo_id, split="train")
            existing_df = existing_ds.to_pandas()  # type: ignore
        except:
            # Dataset doesn't exist yet or is empty
            pass
        
        # Helper function to convert numpy types to Python types
        def convert_value(val):
            if val is None:
                return None
            if isinstance(val, (np.integer, np.floating)):
                return val.item()
            if isinstance(val, np.ndarray):
                return None  # Skip arrays
            return val
        
        # Flatten metrics for easy comparison
        row_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            
            # Config fields
            "base_model": config.get("base_model"),
            "train_size": config.get("train_size"),
            "test_size": config.get("test_size"),
            "full_train_size": config.get("full_train_size"),
            "batch_size": config.get("batch_size"),
            "num_epochs": config.get("num_epochs"),
            "sample_size": config.get("sample_size"),
            "sampling_strategy": config.get("sampling_strategy"),
            "test_split": config.get("test_split"),
            "num_iterations": config.get("num_iterations"),
            
            # Overall metrics (convert numpy types)
            "f1_micro": convert_value(metrics.get("overall", {}).get("f1_micro")),
            "f1_macro": convert_value(metrics.get("overall", {}).get("f1_macro")),
            "f1_samples": convert_value(metrics.get("overall", {}).get("f1_samples")),
            "accuracy": convert_value(metrics.get("overall", {}).get("accuracy")),
            "hamming_loss": convert_value(metrics.get("overall", {}).get("hamming_loss")),
        }
        
        # Add per-label metrics (convert numpy types)
        per_label = metrics.get("per_label", {})
        for label_name, label_metrics in per_label.items():
            row_data[f"{label_name.lower()}_precision"] = convert_value(label_metrics.get("precision"))
            row_data[f"{label_name.lower()}_recall"] = convert_value(label_metrics.get("recall"))
            row_data[f"{label_name.lower()}_f1"] = convert_value(label_metrics.get("f1"))
            row_data[f"{label_name.lower()}_support"] = convert_value(label_metrics.get("support"))
        
        # Add metadata
        if metadata:
            row_data["model_repo"] = metadata.get("model_repo")
            row_data["dataset_repo"] = metadata.get("dataset_repo")
            row_data["dataset_version"] = metadata.get("dataset_version")
            row_data["notes"] = metadata.get("notes", "")
        
        # Create new dataframe
        new_row_df = pd.DataFrame([row_data])
        
        # Append to existing or create new
        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)  # type: ignore
        else:
            combined_df = new_row_df
        
        # Save to CSV and upload
        csv_path = Path("evals_results.csv")
        combined_df.to_csv(csv_path, index=False)
        
        api.upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo="evals_results.csv",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add evaluation results for {version}"
        )
        
        # Create/update README if it doesn't exist
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except:
            readme_content = create_evals_readme(repo_id)
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Initialize evals dataset README"
            )
        
        # Clean up temporary file
        if csv_path.exists():
            csv_path.unlink()
        
        print(f"  ✓ Uploaded evaluation results to: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to upload evaluation results: {e}")
        return False


def create_evals_readme(repo_id: str) -> str:
    """Create README content for evals dataset."""
    return f"""---
license: mit
task_categories:
- evaluation
language:
- en
tags:
- water-conflict
- experiment-tracking
- evaluation-results
- model-comparison
- setfit
size_categories:
- n<1K
---

# Water Conflict Classifier - Evaluation Results

This dataset tracks evaluation results and hyperparameter configurations across different training runs of the water conflict classifier.

**Purpose**: Compare model performance across experiments to identify optimal training configurations and track improvements over time.

## Dataset Structure

### Files

- `evals_results.csv`: Evaluation metrics and configurations for all training runs

### Columns

#### Identifiers
- `version`: Model version (e.g., v1.0, v1.1)
- `timestamp`: ISO 8601 timestamp of training

#### Configuration
- `base_model`: Base model used for training
- `train_size`: Number of training examples used
- `test_size`: Number of test examples
- `full_train_size`: Total training pool size (before sampling)
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `sample_size`: Number of samples used if sampling was applied
- `sampling_strategy`: Strategy used (e.g., undersampling, oversampling)
- `test_split`: Test set split ratio
- `num_iterations`: SetFit contrastive pair iterations

#### Overall Metrics
- `f1_micro`: Micro-averaged F1 score
- `f1_macro`: Macro-averaged F1 score
- `f1_samples`: Sample-averaged F1 score
- `accuracy`: Exact match accuracy
- `hamming_loss`: Hamming loss

#### Per-Label Metrics
For each label (trigger, casualty, weapon):
- `[label]_precision`: Precision score
- `[label]_recall`: Recall score
- `[label]_f1`: F1 score
- `[label]_support`: Number of true examples in test set

#### Metadata
- `model_repo`: HF model repository
- `dataset_repo`: HF training data repository
- `dataset_version`: Training dataset version (e.g., d1.0, d1.1)
- `notes`: Optional notes about the training run

## Usage

```python
from datasets import load_dataset
import pandas as pd

# Load evaluation results
dataset = load_dataset("{repo_id}")
df = dataset['train'].to_pandas()

# Compare different configurations
df.sort_values('f1_micro', ascending=False).head(10)

# Analyze impact of sample size on performance
df.plot.scatter(x='sample_size', y='f1_micro')

# Compare different epochs
df.groupby('num_epochs')['f1_micro'].mean()
```

## Purpose

This dataset enables:
- Tracking model performance over time
- Comparing hyperparameter configurations
- Identifying optimal training settings
- Reproducing best-performing models
"""


def compare_experiments_from_hf(
    repo_id: str,
    version1: str,
    version2: str,
    token: Optional[str] = None
) -> dict:
    """
    Compare two experiments from HF evals dataset.
    
    Args:
        repo_id: HF dataset repo ID
        version1: First version to compare
        version2: Second version to compare
        token: Optional HF token
        
    Returns:
        Dictionary with comparison results
    """
    try:
        from datasets import load_dataset
        
        ds = load_dataset(repo_id, token=token, split="train")
        df = ds.to_pandas()  # type: ignore
        
        exp1 = df[df['version'] == version1].iloc[0].to_dict()  # type: ignore
        exp2 = df[df['version'] == version2].iloc[0].to_dict()  # type: ignore
        
        # Compare key metrics
        metrics_to_compare = ['f1_micro', 'f1_macro', 'accuracy']
        comparison = {
            "version1": version1,
            "version2": version2,
            "differences": {}
        }
        
        for metric in metrics_to_compare:
            v1_val = exp1.get(metric)
            v2_val = exp2.get(metric)
            if v1_val is not None and v2_val is not None:
                diff = v2_val - v1_val
                pct_change = (diff / v1_val) * 100 if v1_val != 0 else 0
                comparison["differences"][metric] = {
                    "v1": v1_val,
                    "v2": v2_val,
                    "diff": diff,
                    "pct_change": pct_change
                }
        
        return comparison
        
    except Exception as e:
        return {"error": str(e)}


def get_best_experiments(
    repo_id: str,
    metric: str = "f1_micro",
    top_n: int = 5,
    token: Optional[str] = None
) -> pd.DataFrame:
    """
    Get top N experiments by metric from HF evals dataset.
    
    Args:
        repo_id: HF dataset repo ID
        metric: Metric to sort by (default: f1_micro)
        top_n: Number of top experiments to return
        token: Optional HF token
        
    Returns:
        DataFrame with top N experiments
    """
    try:
        from datasets import load_dataset
        
        ds = load_dataset(repo_id, token=token, split="train")
        df = ds.to_pandas()  # type: ignore
        
        return df.sort_values(metric, ascending=False).head(top_n)  # type: ignore
        
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return pd.DataFrame()

