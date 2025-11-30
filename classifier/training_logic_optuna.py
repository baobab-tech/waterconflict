"""
Optuna-based hyperparameter search training logic for SetFit water conflict classifier.

This module provides an alternative training approach using Optuna for hyperparameter
optimization. Based on the approach from: https://www.philschmid.de/setfit-outperforms-gpt3

Key differences from standard training:
- Uses model_init function for hyperparameter search
- Searches across learning rates, epochs, batch sizes, iterations
- Optionally searches across multiple base models
- Uses SMALL SAMPLE for fast hyperparameter search, then trains final model on full data

The key insight: SetFit is designed for few-shot learning. Using 1200 samples for each
Optuna trial is wasteful. Instead, we:
1. Sample ~200 examples for hyperparameter search (fast trials)
2. Find best hyperparameters across N trials
3. Train final model on FULL dataset with best hyperparameters

Usage:
    from training_logic_optuna import train_model_with_optuna

    model, best_params = train_model_with_optuna(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        n_trials=50,
        search_sample_size=200,  # Small sample for fast search
        label_names=['Trigger', 'Casualty', 'Weapon']
    )
"""

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, SetFitModelCardData
from typing import Optional
import numpy as np
import optuna


# Default base models to search across
DEFAULT_BASE_MODELS = [
    "BAAI/bge-small-en-v1.5",      # 33M params, fast
    "BAAI/bge-base-en-v1.5",       # 110M params, better quality
    "sentence-transformers/all-mpnet-base-v2",  # 110M params, battle-tested
]


def _stratified_sample_multilabel(dataset: Dataset, sample_size: int, seed: int = 42) -> Dataset:
    """
    Create a stratified sample from a multi-label dataset.

    Ensures all labels are represented proportionally in the sample.

    Args:
        dataset: HF Dataset with 'labels' column containing lists like [1,0,1]
        sample_size: Target number of samples
        seed: Random seed for reproducibility

    Returns:
        Sampled Dataset
    """
    if len(dataset) <= sample_size:
        return dataset

    np.random.seed(seed)

    # Get all labels
    labels = np.array(dataset['labels'])
    n_labels = labels.shape[1]

    # Calculate samples per label (with overlap allowed for multi-label)
    samples_per_label = sample_size // n_labels

    selected_indices = set()

    # Sample for each label to ensure representation
    for label_idx in range(n_labels):
        # Find indices where this label is positive
        label_positive_indices = np.where(labels[:, label_idx] == 1)[0]

        if len(label_positive_indices) > 0:
            n_to_sample = min(len(label_positive_indices), samples_per_label)
            sampled = np.random.choice(label_positive_indices, size=n_to_sample, replace=False)
            selected_indices.update(sampled.tolist())

    # Add negatives if we haven't reached target
    if len(selected_indices) < sample_size:
        # Find negative indices (all zeros)
        negative_indices = np.where(labels.sum(axis=1) == 0)[0]
        remaining = sample_size - len(selected_indices)

        available_negatives = [i for i in negative_indices if i not in selected_indices]
        if available_negatives:
            n_neg_sample = min(len(available_negatives), remaining)
            neg_sampled = np.random.choice(available_negatives, size=n_neg_sample, replace=False)
            selected_indices.update(neg_sampled.tolist())

    # If still under target, add random samples
    if len(selected_indices) < sample_size:
        remaining = sample_size - len(selected_indices)
        available = [i for i in range(len(dataset)) if i not in selected_indices]
        if available:
            extra = np.random.choice(available, size=min(len(available), remaining), replace=False)
            selected_indices.update(extra.tolist())

    return dataset.select(list(selected_indices))


def train_model_with_optuna(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    label_names: list[str] = None,
    n_trials: int = 50,
    search_sample_size: int = 200,
    direction: str = "maximize",
    metric: str = "f1",
    base_models: list[str] = None,
    search_models: bool = False,
    model_card_data: SetFitModelCardData = None,
    timeout: int = None,
) -> tuple[SetFitModel, dict]:
    """
    Train SetFit multi-label classifier with Optuna hyperparameter search.

    Strategy:
    1. Create a SMALL stratified sample for hyperparameter search (fast trials)
    2. Run N trials on the small sample to find best hyperparameters
    3. Train final model on FULL dataset with best hyperparameters

    This is much faster than searching on the full dataset, and SetFit's few-shot
    nature means hyperparameters found on ~200 samples transfer well to larger data.

    Args:
        train_dataset: HF Dataset with 'text' and 'labels' columns (full data)
        eval_dataset: HF Dataset with 'text' and 'labels' columns
        label_names: List of label names (e.g. ['Trigger', 'Casualty', 'Weapon'])
        n_trials: Number of Optuna trials to run (default 50)
        search_sample_size: Number of samples to use for hyperparameter search (default 200)
        direction: 'maximize' or 'minimize' for optimization
        metric: Metric to optimize ('f1', 'accuracy', etc.)
        base_models: List of base models to search across (if search_models=True)
        search_models: Whether to search across different base models
        model_card_data: Optional SetFitModelCardData for model card metadata
        timeout: Optional timeout in seconds for the entire search

    Returns:
        Tuple of (best_model, best_hyperparameters)
    """
    if label_names is None:
        label_names = ['Trigger', 'Casualty', 'Weapon']

    if base_models is None:
        base_models = DEFAULT_BASE_MODELS

    print(f"\n  Optuna Hyperparameter Search")
    print(f"  ============================")
    print(f"  Full training data: {len(train_dataset)} samples")
    print(f"  Search sample size: {search_sample_size} samples")
    print(f"  Trials: {n_trials}")
    print(f"  Metric: {metric} ({direction})")
    print(f"  Search models: {search_models}")
    if search_models:
        print(f"  Candidate models: {base_models}")
    else:
        print(f"  Base model: {base_models[0]}")
    print(f"  Strategy: One-vs-Rest multi-label classification")

    # Create stratified sample for fast hyperparameter search
    print(f"\n  Creating stratified sample for search...")
    search_train_dataset = _stratified_sample_multilabel(
        train_dataset, search_sample_size, seed=42
    )
    print(f"  ✓ Search sample: {len(search_train_dataset)} examples")

    # Model initialization function for hyperparameter search
    def model_init(params: dict = None) -> SetFitModel:
        params = params or {}

        # LogisticRegression head parameters
        max_iter = params.get("max_iter", 100)
        solver = params.get("solver", "liblinear")

        # Model selection
        if search_models:
            model_id = params.get("model_id", base_models[0])
        else:
            model_id = base_models[0]

        model_params = {
            "head_params": {
                "max_iter": max_iter,
                "solver": solver,
                "class_weight": "balanced",  # Always use balanced for our imbalanced data
            }
        }

        return SetFitModel.from_pretrained(
            model_id,
            multi_target_strategy="one-vs-rest",
            labels=label_names,
            model_card_data=model_card_data,
            **model_params
        )

    # Hyperparameter search space
    def hp_space(trial: optuna.Trial) -> dict:
        hp = {
            # Training hyperparameters
            "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-4, log=True),
            "head_learning_rate": trial.suggest_float("head_learning_rate", 1e-3, 1e-1, log=True),
            "num_epochs": trial.suggest_int("num_epochs", 1, 5),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_iterations": trial.suggest_categorical("num_iterations", [10, 20, 40]),
            "seed": trial.suggest_int("seed", 1, 100),

            # LogisticRegression head hyperparameters
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),

            # Sampling strategy
            "sampling_strategy": trial.suggest_categorical("sampling_strategy", ["oversampling", "undersampling"]),
        }

        # Optionally search across models
        if search_models:
            hp["model_id"] = trial.suggest_categorical("model_id", base_models)

        return hp

    # Create trainer for hyperparameter search using SAMPLED data (fast)
    search_trainer = Trainer(
        train_dataset=search_train_dataset,  # Use small sample for search
        eval_dataset=eval_dataset,
        model_init=model_init,
        metric=metric,
        column_mapping={"text": "text", "labels": "label"}
    )

    print(f"\n  Starting hyperparameter search on {len(search_train_dataset)} samples...")
    print(f"  This is much faster than searching on full {len(train_dataset)} samples.\n")

    # Run hyperparameter search on sampled data
    best_run = search_trainer.hyperparameter_search(
        direction=direction,
        hp_space=hp_space,
        n_trials=n_trials,
        timeout=timeout,
    )

    print(f"\n  ✓ Hyperparameter search complete!")
    print(f"\n  Best hyperparameters (found on {len(search_train_dataset)} samples):")
    for key, value in best_run.hyperparameters.items():
        print(f"    {key}: {value}")
    print(f"\n  Best {metric} on search sample: {best_run.objective:.4f}")

    # Now train final model on FULL dataset with best hyperparameters
    print(f"\n  Training final model on FULL dataset ({len(train_dataset)} samples)...")
    print(f"  Using best hyperparameters from search...")

    # Create new trainer with FULL dataset
    final_trainer = Trainer(
        train_dataset=train_dataset,  # Use FULL data for final model
        eval_dataset=eval_dataset,
        model_init=model_init,
        metric=metric,
        column_mapping={"text": "text", "labels": "label"}
    )

    final_trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    final_trainer.train()

    print(f"\n  ✓ Final model training complete (trained on {len(train_dataset)} samples)!")

    return final_trainer.model, best_run.hyperparameters


def train_model_quick_search(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    label_names: list[str] = None,
    base_model: str = "BAAI/bge-small-en-v1.5",
    n_trials: int = 20,
    model_card_data: SetFitModelCardData = None,
) -> tuple[SetFitModel, dict]:
    """
    Quick hyperparameter search with reduced search space.

    Good for initial exploration or when compute is limited.
    Searches only the most impactful hyperparameters.

    Args:
        train_dataset: HF Dataset with 'text' and 'labels' columns
        eval_dataset: HF Dataset with 'text' and 'labels' columns
        label_names: List of label names
        base_model: Base model to use (no model search)
        n_trials: Number of trials (default 20)
        model_card_data: Optional SetFitModelCardData

    Returns:
        Tuple of (best_model, best_hyperparameters)
    """
    if label_names is None:
        label_names = ['Trigger', 'Casualty', 'Weapon']

    print(f"\n  Quick Hyperparameter Search")
    print(f"  ===========================")
    print(f"  Trials: {n_trials}")
    print(f"  Base model: {base_model}")

    def model_init(params: dict = None) -> SetFitModel:
        params = params or {}
        return SetFitModel.from_pretrained(
            base_model,
            multi_target_strategy="one-vs-rest",
            labels=label_names,
            model_card_data=model_card_data,
            head_params={"class_weight": "balanced"}
        )

    def hp_space(trial: optuna.Trial) -> dict:
        return {
            "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-4, log=True),
            "num_epochs": trial.suggest_int("num_epochs", 1, 4),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "num_iterations": trial.suggest_categorical("num_iterations", [10, 20]),
            "sampling_strategy": trial.suggest_categorical("sampling_strategy", ["oversampling", "undersampling"]),
        }

    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
        metric="f1",
        column_mapping={"text": "text", "labels": "label"}
    )

    print(f"\n  Starting quick search...")

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        n_trials=n_trials,
    )

    print(f"\n  ✓ Search complete! Best F1: {best_run.objective:.4f}")
    print(f"  Best params: {best_run.hyperparameters}")

    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()

    return trainer.model, best_run.hyperparameters
