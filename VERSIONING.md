# Versioning & Experiment Tracking

This project uses a dual-versioning system for complete reproducibility:
- **Dataset versions** (`d1.0`, `d1.1`, ...) - track training data snapshots
- **Model versions** (`v1.0`, `v1.1`, ...) - track trained models
- Each model training automatically records which dataset version was used

## Version Formats

- **Dataset**: `d1.0`, `d1.1`, `d2.0` (prefix: `d`)
- **Model**: `v1.0`, `v1.1`, `v2.0` (prefix: `v`)

This allows independent tracking (e.g., model `v2.3` trained on dataset `d1.5`).

## Features

- **Dual Version Tracking**: Separate versions for datasets and models with automatic linking
- **Experiment History**: Logs all configs, metrics, and dataset→model mappings to `experiment_history.jsonl`
- **HuggingFace Hub Tags**: Git tags on HF Hub for both datasets and models
- **Evaluation Dataset**: Cross-experiment comparison on HF Hub
- **Comparison Tools**: Compare metrics across versions (`view_experiments.py`, `view_evals.py`)

## Quick Start

### 1. Prepare Training Dataset (with versioning)

```shell
# First time - creates d1.0
python scripts/prepare_training_dataset.py

# Minor update (d1.0 → d1.1)
python scripts/prepare_training_dataset.py

# Major update (d1.5 → d2.0)
python scripts/prepare_training_dataset.py --major
```

### 2. Train Model (auto-detects dataset version)

```shell
# Trains model, auto-detects dataset version (d1.0, d1.1, etc.)
uv run scripts/train_on_hf.py

# Creates model version (v1.0, v1.1, etc.)
# Records which dataset version was used
```

### 3. View Experiment History

```shell
# View recent experiments (shows dataset version used)
python scripts/view_experiments.py

# Compare two model versions
python scripts/view_experiments.py --compare v1.0 v1.2

# View from HF Hub evals dataset
python scripts/view_evals.py
```

Example output:
```
================================================================================
EXPERIMENT HISTORY (last 5)
================================================================================

v1.2 - 2024-11-26T14:30:45
  Config:
    Training samples: 1200
    Dataset version: d1.0  ← Which data was used
    Base model: BAAI/bge-small-en-v1.5
    Epochs: 1
  Metrics:
    F1 (micro): 0.8765
    Accuracy: 0.8234
    Trigger F1: 0.9123
```

## Why Separate Dataset & Model Versions?

**Compare fairly**: Compare models trained on same dataset
```shell
v1.0 on d1.0: F1=0.85
v1.1 on d1.0: F1=0.87  ← Better hyperparameters
```

**Track data impact**: See how data changes affect performance
```shell
v2.0 on d1.0: F1=0.87
v2.1 on d2.0: F1=0.90  ← Better data quality
```

**Reproduce exactly**: Re-train with same data by referencing dataset version

## How It Works

### Dataset Preparation (`prepare_training_dataset.py`)
1. Preprocesses and samples data
2. Uploads to `{org}/water-conflict-training-data`
3. Creates git tag (e.g., `d1.0`)
4. Increments automatically: `d1.0` → `d1.1` → `d2.0`

### Model Training (`train_on_hf.py`)
1. Loads training data from HF Hub
2. Auto-detects dataset version from tags (finds latest: `d1.0`, `d1.1`, etc.)
3. Trains model
4. Creates model version tag (e.g., `v1.0`)
5. Records dataset→model mapping everywhere:
   - `experiment_history.jsonl` (local)
   - Model card on HF Hub
   - Evals dataset on HF Hub

## Loading Specific Versions

```python
from setfit import SetFitModel
from datasets import load_dataset

# Load specific model version
model = SetFitModel.from_pretrained("org/water-conflict-classifier", revision="v1.2")

# Load specific dataset version
dataset = load_dataset("org/water-conflict-training-data", revision="d1.0")
```

## Manual Version Control

```shell
# Dataset version
python scripts/prepare_training_dataset.py --version d2.0
python scripts/prepare_training_dataset.py --major  # d1.5 → d2.0

# Model version
MODEL_VERSION=v2.0 uv run scripts/train_on_hf.py
VERSION_BUMP=major uv run scripts/train_on_hf.py  # v1.5 → v2.0
```

## Experiment History Format

Logged to `experiment_history.jsonl`:

```json
{
  "version": "v1.2",
  "timestamp": "2024-11-26T14:30:45",
  "config": {
    "dataset_version": "d1.0",  ← Links to dataset
    "base_model": "BAAI/bge-small-en-v1.5",
    "train_size": 1200,
    "batch_size": 32,
    "num_epochs": 1
  },
  "metrics": {
    "overall": {"f1_micro": 0.8765, "accuracy": 0.8234}
  },
  "metadata": {
    "dataset_version": "d1.0",
    "training_dataset_repo": "org/water-conflict-training-data"
  }
}
```

## Viewing & Comparing

### Local History
```shell
# View recent experiments with dataset versions
python scripts/view_experiments.py

# Compare two model versions
python scripts/view_experiments.py --compare v1.0 v1.2
```

### HF Hub Evals
```shell
# View all experiments from HF Hub
python scripts/view_evals.py

# Show top performers
python scripts/view_evals.py --top 10

# Compare versions
python scripts/view_evals.py --compare v1.0 v1.2
```

Both tools now show which dataset version (`d1.0`, `d1.1`, etc.) was used for each model.

## Where Versions Are Tracked

1. **HF Hub Tags**:
   - Dataset repo: `d1.0`, `d1.1`, `d2.0`, ...
   - Model repo: `v1.0`, `v1.1`, `v2.0`, ...

2. **Experiment History** (`experiment_history.jsonl`):
   - Records dataset_version in config
   - Local tracking of all experiments

3. **Model Card**: Shows dataset version used

4. **Evals Dataset** (`{org}/water-conflict-classifier-evals`):
   - All experiments with dataset→model mapping
   - Queryable via `view_evals.py`

## Example Workflow

```shell
# 1. Prepare dataset (creates d1.0)
python scripts/prepare_training_dataset.py

# 2. Train model (creates v1.0, records it used d1.0)
uv run scripts/train_on_hf.py

# 3. View results
python scripts/view_experiments.py
# Shows: v1.0 trained on d1.0, F1=0.85

# 4. Update dataset (creates d1.1)
python scripts/prepare_training_dataset.py --sample-size 1500

# 5. Train again (creates v1.1, records it used d1.1)
uv run scripts/train_on_hf.py

# 6. Compare
python scripts/view_experiments.py --compare v1.0 v1.1
# Shows: v1.0 (d1.0) vs v1.1 (d1.1) - data size impact
```

## Best Practices

- **Let it auto-version**: Manual versions only when needed
- **Dataset changes → new dataset version**: Re-run `prepare_training_dataset.py`
- **Review before training**: Check recent experiments first
- **Major versions** for big changes: `--major` flag

