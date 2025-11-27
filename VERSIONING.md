# Model Versioning & Experiment Tracking

This project includes a built-in versioning system for tracking model experiments and comparing performance across training runs.

## Features

- **Automatic Version Numbering**: Auto-generates version numbers (v1.0, v1.1, etc.)
- **Experiment History**: Logs all training configs and metrics to `experiment_history.jsonl`
- **Versioned Training Datasets**: Creates a separate HF dataset for each training run showing exact sampled data used
- **HuggingFace Hub Tags**: Creates git tags on HF Hub for easy model version retrieval
- **Evaluation Dataset**: Uploads evaluation results to HF for cross-experiment comparison
- **Comparison Tools**: Compare metrics across different versions
- **Extensible**: Easy to extend to W&B or MLflow later

## Quick Start

### Training with Auto-Versioning

Both training scripts automatically log experiments:

```bash
# Local training - auto-logs to experiment_history.jsonl
cd classifier
python train_setfit_headline_classifier.py

# HF Jobs training - logs + creates HF Hub tag
cd scripts
uv run train_on_hf.py
```

### Viewing Experiment History

```bash
# View last 5 experiments
python scripts/view_experiments.py

# View all experiments
python scripts/view_experiments.py --all

# View last 10 experiments
python scripts/view_experiments.py --limit 10
```

Example output:
```
================================================================================
EXPERIMENT HISTORY (last 5)
================================================================================

v1.2 - 2024-11-26T14:30:45
  Config:
    Training samples: 1200
    Base model: BAAI/bge-small-en-v1.5
    Epochs: 1
  Metrics:
    F1 (micro): 0.8765
    Accuracy: 0.8234
    Trigger F1: 0.9123
    Casualty F1: 0.8845
    Weapon F1: 0.8327
```

### Comparing Versions

```bash
# Compare two versions
python scripts/view_experiments.py --compare v1.0 v1.2

# Compare using different metric
python scripts/view_experiments.py --compare v1.0 v1.2 --metric accuracy
```

Example output:
```
Comparing v1.0 vs v1.2
================================================================================

v1.0: f1_micro = 0.8234
v1.2: f1_micro = 0.8765

Difference: +0.0531 (+6.45%)
✓ Improvement detected
```

## HuggingFace Hub Integration

### Automatic Tagging & Dataset Versioning

When training with `train_on_hf.py`, the script automatically:
1. Uploads versioned training dataset (e.g., `org/water-conflict-training-data-v1.0`)
2. Trains model and pushes to HF Hub
3. Creates a git tag on your HF Hub model repository
4. Uploads evaluation results to evals dataset
5. Logs experiment to `experiment_history.jsonl`
6. Links all artifacts together (model → training dataset → evals)

### Loading Specific Versions

```python
from setfit import SetFitModel
from datasets import load_dataset

# Load latest model
model = SetFitModel.from_pretrained("your-org/water-conflict-classifier")

# Load specific model version by tag
model_v10 = SetFitModel.from_pretrained(
    "your-org/water-conflict-classifier",
    revision="v1.0"
)

model_v12 = SetFitModel.from_pretrained(
    "your-org/water-conflict-classifier", 
    revision="v1.2"
)

# Load the exact training data used for a specific version
training_data_v10 = load_dataset("your-org/water-conflict-training-data-v1.0")
train_split = training_data_v10['train']  # Training samples
test_split = training_data_v10['test']    # Test samples
```

### Manual Version Specification

Override auto-versioning by setting `MODEL_VERSION`:

```bash
# Specify exact version
export MODEL_VERSION=v2.0
uv run scripts/train_on_hf.py

# Or inline
MODEL_VERSION=v1.5 uv run scripts/train_on_hf.py
```

## Experiment History Format

Experiments are logged to `experiment_history.jsonl` in JSON Lines format:

```json
{
  "version": "v1.0",
  "timestamp": "2024-11-26T14:30:45.123456",
  "config": {
    "base_model": "BAAI/bge-small-en-v1.5",
    "train_size": 1200,
    "test_size": 180,
    "batch_size": 32,
    "num_epochs": 1,
    "sampling_strategy": "undersampling"
  },
  "metrics": {
    "overall": {
      "f1_micro": 0.8765,
      "f1_macro": 0.8543,
      "accuracy": 0.8234
    },
    "per_label": {
      "Trigger": {"f1": 0.9123, "precision": 0.9234, "recall": 0.9015},
      "Casualty": {"f1": 0.8845, "precision": 0.8923, "recall": 0.8768},
      "Weapon": {"f1": 0.8327, "precision": 0.8456, "recall": 0.8201}
    }
  },
  "metadata": {
    "model_repo": "baobabtech/water-conflict-classifier",
    "dataset_repo": "baobabtech/water-conflict-training-data",
    "training_dataset_repo": "baobabtech/water-conflict-training-data-v1.0"
  }
}
```

## Versioned Training Datasets

### What Gets Created

Each training run creates a separate versioned dataset on HF Hub containing:
- **train.csv**: The exact sampled training data used
- **test.csv**: The held-out test set
- **README.md**: Dataset card with sampling configuration and metadata

### Dataset Naming

Versioned datasets follow the pattern: `{org}/{base-dataset-name}-v{version}`

Example:
- Base dataset: `baobabtech/water-conflict-training-data`
- Model version: `v1.0`
- Training dataset: `baobabtech/water-conflict-training-data-v1.0`

### Why This Matters

**Full Reproducibility**: Anyone can see exactly what data trained a specific model version, including:
- Which samples were selected during stratified sampling
- Train/test split composition
- Sampling configuration parameters

**Transparency**: Clear audit trail from raw data → sampled data → trained model → evaluation results

**Comparison**: Compare not just model performance, but the actual training data used across versions

### Accessing Training Datasets

```python
from datasets import load_dataset

# Load training dataset for a specific model version
dataset = load_dataset("your-org/water-conflict-training-data-v1.0")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

# View as pandas
import pandas as pd
train_df = train_data.to_pandas()
test_df = test_data.to_pandas()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
```

### Dataset Card Contents

Each versioned dataset includes:
- Parent dataset reference
- Model version link
- Exact sample counts (train/test)
- Sampling configuration (sample size, stratification settings, etc.)
- Label distribution
- Usage instructions

## Programmatic Usage

### Using the ExperimentTracker

```python
from versioning import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker("experiment_history.jsonl")

# Log an experiment
tracker.log_experiment(
    version="v1.5",
    config={"train_size": 1500, "epochs": 2},
    metrics={"f1_micro": 0.89},
    metadata={"notes": "Increased training data"}
)

# Get all experiments
experiments = tracker.get_experiments()

# Get specific experiment
exp = tracker.get_experiment_by_version("v1.5")

# Compare two experiments
comparison = tracker.compare_experiments("v1.0", "v1.5")
print(comparison['percent_change'])  # e.g., +8.45

# Print summary
tracker.print_summary(limit=10)
```

### Manual HF Hub Tagging

```python
from versioning import create_hf_version_tag

# Create a tag on HF Hub
create_hf_version_tag(
    model_repo="your-org/water-conflict-classifier",
    version="v1.5",
    metrics=eval_results,
    config=training_config
)
```

### Version Management

```python
from versioning import get_next_version

# Get next minor version (v1.0 -> v1.1)
next_version = get_next_version("experiment_history.jsonl")

# Get next major version (v1.5 -> v2.0)
next_major = get_next_version("experiment_history.jsonl", major=True)
```

## Integration with W&B or MLflow (Future)

The current system is designed to be extended easily:

```python
# Example W&B integration (future)
import wandb
from versioning import ExperimentTracker

wandb.init(project="water-conflict-classifier", name=version)
wandb.config.update(config)

# Train model...

wandb.log(eval_results['overall'])
wandb.log_artifact(model_path, type="model")

# Still log locally for consistency
tracker.log_experiment(version, config, metrics)
wandb.finish()
```

## Best Practices

1. **Let it auto-version**: Don't manually specify versions unless needed
2. **Review before training**: Check recent experiments with `view_experiments.py`
3. **Compare after training**: Always compare new version to previous best
4. **Tag releases**: Use major versions (v2.0, v3.0) for production releases
5. **Document changes**: Add notes in metadata for significant changes

## File Locations

- **Experiment history**: `experiment_history.jsonl` (project root)
- **Versioning module**: `classifier/versioning.py`
- **View utility**: `scripts/view_experiments.py`
- **HF tags**: On your model repository at https://huggingface.co/your-org/model/tags

## Troubleshooting

### Version conflicts

If you see duplicate versions, specify explicit version:
```bash
MODEL_VERSION=v1.5-retry uv run scripts/train_on_hf.py
```

### Missing history file

The file is created automatically on first training run. If deleted, versioning starts fresh at v1.0.

### HF Hub tag failures

- Check authentication: `huggingface-cli login`
- Verify repository exists and you have write access
- Check if tag already exists (tags must be unique)

## Example Workflow

```bash
# 1. Train initial model
python classifier/train_setfit_headline_classifier.py
# → Logs as v1.0

# 2. View results
python scripts/view_experiments.py
# → Shows v1.0 metrics

# 3. Adjust hyperparameters and retrain
# Edit train_setfit_headline_classifier.py (e.g., increase epochs)
python classifier/train_setfit_headline_classifier.py
# → Logs as v1.1

# 4. Compare versions
python scripts/view_experiments.py --compare v1.0 v1.1
# → Shows improvement (or not)

# 5. Push to HF Hub with versioning
uv run scripts/train_on_hf.py
# → Creates versioned training dataset (org/water-conflict-training-data-v1.2)
# → Trains and uploads model
# → Logs as v1.2 + creates HF Hub tag
# → Uploads evals to comparison dataset

# 6. Later, load specific version and its training data
python
>>> from setfit import SetFitModel
>>> from datasets import load_dataset
>>> model = SetFitModel.from_pretrained("org/model", revision="v1.2")
>>> training_data = load_dataset("org/water-conflict-training-data-v1.2")
```

## Next Steps

- Add visualization dashboard (plotly/streamlit) for experiment comparison
- Integrate with Weights & Biases for real-time charts
- Add automatic model selection (load best performing version)
- Export comparison tables to markdown/CSV
- Add dataset diff tool (compare training data across versions)
- Automated data quality checks for training datasets

