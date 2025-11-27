# Utility Scripts

Scripts for data preparation, dataset management, cloud training, and experiment tracking. These scripts use the published [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/) package.

## Scripts

### `view_experiments.py`
View and compare model training experiments from local experiment history. All training runs are automatically logged with metrics and configs.

**Usage:**
```bash
# View recent experiments
python scripts/view_experiments.py

# View all experiments
python scripts/view_experiments.py --all

# Compare two versions
python scripts/view_experiments.py --compare v1.0 v1.1

# Compare using different metric
python scripts/view_experiments.py --compare v1.0 v1.2 --metric accuracy
```

**What it does:**
- Displays experiment history from `experiment_history.jsonl`
- Shows training configs, metrics, and timestamps
- Compares performance between versions
- Helps identify improvements and regressions

**See:** `../VERSIONING.md` for full documentation

---

### `view_evals.py`
View and compare evaluation results from HuggingFace evals dataset. Training runs automatically upload results to HF for easy comparison across experiments.

**Usage:**
```bash
# View all experiments sorted by F1 score
python scripts/view_evals.py

# Show top 10 experiments
python scripts/view_evals.py --top 10

# Compare two specific versions
python scripts/view_evals.py --compare v1.0 v1.1

# Analyze impact of hyperparameters
python scripts/view_evals.py --analyze sample_size
python scripts/view_evals.py --analyze num_epochs

# Show detailed per-label metrics for top experiments
python scripts/view_evals.py --labels --top 5
```

**What it does:**
- Loads evaluation results from HF dataset (`YOUR_ORG/water-conflict-classifier-evals`)
- Displays metrics, configs, and comparisons in tabular format
- Analyzes hyperparameter impact on performance
- Shows per-label precision, recall, F1 for each experiment
- Helps identify optimal training configurations

**Requirements:**
- HF authentication: `hf auth login`
- Evals dataset populated via `train_on_hf.py` runs

---

### `demo_versioning.py`
Create sample experiments to demonstrate the versioning system.

**Usage:**
```bash
python scripts/demo_versioning.py
```

Creates demo data in `demo_experiment_history.jsonl` to explore the versioning features without running actual training.

---

### `transform_prep_negatives.py`
Generate negative examples (non-water conflict headlines) from ACLED data.

**Usage:**
```bash
cd scripts
python transform_prep_negatives.py
```

**What it does:**
- Loads ACLED raw data from `../data/ACLED RAW_*.csv`
- Filters out headlines containing "water"
- Samples 2,500 random headlines
- Cleans headlines (removes date prefixes)
- Saves to `../data/negatives.csv`

**Requirements:** 
- ACLED data must be in `../data/` folder

**Note:** This creates the base ACLED negatives pool. Run `generate_hard_negatives.py` after this to create the final training-ready dataset with hard negatives included.

---

### `generate_hard_negatives.py`
Generate "hard negatives" - peaceful water-related news to prevent false positives.

**Usage:**
```bash
cd scripts
python generate_hard_negatives.py
```

**What it does:**
- Generates ~120 peaceful water-related headlines (infrastructure, research, conservation)
- Creates `../data/hard_negatives.csv`
- Merges with ACLED negatives to create training-ready `negatives_updated.csv`
- Tags hard negatives with `priority_sample=True` to ensure they're always included
- Samples down ACLED negatives to ~600 (since we don't need all 2500+)

**Why needed:** Without hard negatives, the model learns "water = conflict" instead of "water + violence = conflict". Hard negatives teach it to distinguish peaceful water news from actual conflicts.

**Output:** Creates `negatives_updated.csv` with:
- ALL 120 hard negatives (priority_sample=True)
- ~600 sampled ACLED negatives (priority_sample=False)
- Total: ~720 training-ready negatives with ~17% hard negatives

---

### `upload_datasets.py`
Upload training datasets to Hugging Face Hub.

**Usage:**
```bash
cd scripts
python upload_datasets.py
```

**What it does:**
- Loads `../data/positives.csv`
- Checks for `../data/negatives_updated.csv` (with hard negatives), falls back to `negatives.csv` if not found
- Creates/updates HF Hub dataset repository
- Uploads files to `YOUR_ORG/water-conflict-training-data`
- Shows dataset composition (hard negatives vs ACLED negatives)

**Requirements:**
- HF authentication: `hf auth login`
- Config file: Copy `config.sample.py` to `config.py` and set `HF_ORGANIZATION`
- Training data must be prepared first (run `generate_hard_negatives.py` for best results)

---

### `train_on_hf.py`
Train the water conflict classifier on HuggingFace Jobs (managed GPUs).

**Usage:**
```bash
# From repo root
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=your-org \
  --namespace your-org \
  scripts/train_on_hf.py

# Optional: specify version explicitly
MODEL_VERSION=v2.0 hf jobs uv run ... scripts/train_on_hf.py
```

**What it does:**
- Downloads training data from HF Hub
- Trains SetFit model on GPU infrastructure
- Evaluates on held-out test set
- Pushes trained model to HF Hub with model card
- Auto-versions and logs experiment to `experiment_history.jsonl`
- Creates git tag on HF Hub for version retrieval
- Uploads evaluation results to HF evals dataset for comparison

**Requirements:**
- Package published to PyPI (already done: [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/))
- HF authentication token (set as `--secrets HF_TOKEN`)
- Training data uploaded to HF Hub (see `upload_datasets.py`)
- Config file with `HF_ORGANIZATION` set

**Configuration Options:**
- `--flavor a10g-large`: GPU type (recommend A10G for this task)
- `--timeout 2h`: Max runtime before auto-termination
- `--secrets HF_TOKEN`: Authentication token
- `--env HF_ORGANIZATION=yourorg`: Your HF org/username
- `--namespace yourorg`: Runs job under org account for billing

**Expected Runtime:** ~2-5 minutes on A10G GPU

**Monitoring:**
```bash
# List jobs
hf jobs ps -a --namespace yourorg

# Stream logs
hf jobs logs <job_id> --namespace yourorg

# Cancel job
hf jobs cancel <job_id> --namespace yourorg
```

**Local Testing:**
```bash
# Test locally before submitting to HF Jobs
uv run scripts/train_on_hf.py
```

Note: Still requires dataset on HF Hub and proper authentication.

---

## Typical Workflow

### For Local Training

```bash
# 1. Generate negative examples
cd scripts
python transform_prep_negatives.py

# 2. Train locally
cd ../classifier
python train_setfit_headline_classifier.py
```

### For Cloud Training (HF Jobs)

```bash
# 1. Generate base ACLED negatives (if needed)
cd scripts
python transform_prep_negatives.py

# 2. Generate training-ready negatives with hard negatives
python generate_hard_negatives.py
# Answer 'y' when prompted to merge - creates negatives_updated.csv

# 3. Configure your HF organization
cd ..
cp config.sample.py config.py
# Edit config.py and set HF_ORGANIZATION

# 4. Upload training data to HF Hub
cd scripts
python upload_datasets.py
# This will use negatives_updated.csv automatically

# 5. Run training on HF Jobs
cd ..
hf jobs uv run \
  --flavor a10g-large \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ORGANIZATION=yourorg \
  --namespace yourorg \
  scripts/train_on_hf.py
```

## Note

These scripts use the published [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/) package. For package source code and local training, see `../classifier/`
