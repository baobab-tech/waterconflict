# Utility Scripts

Scripts for data preparation, dataset management, cloud training, and experiment tracking. These scripts use the published [water-conflict-classifier](https://pypi.org/project/water-conflict-classifier/) package.

## Scripts

### `view_experiments.py`
View and compare model training experiments. All training runs are automatically logged with metrics and configs.

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

---

### `upload_datasets.py`
Upload training datasets to Hugging Face Hub.

**Usage:**
```bash
cd scripts
python upload_datasets.py
```

**What it does:**
- Loads `../data/positives.csv` and `../data/negatives.csv`
- Creates/updates HF Hub dataset repository
- Uploads both files to `YOUR_ORG/water-conflict-training-data`

**Requirements:**
- HF authentication: `hf auth login`
- Config file: Copy `config.sample.py` to `config.py` and set `HF_ORGANIZATION`
- Training data must be prepared first

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
# 1. Generate negative examples (if needed)
cd scripts
python transform_prep_negatives.py

# 2. Configure your HF organization
cd ..
cp config.sample.py config.py
# Edit config.py and set HF_ORGANIZATION

# 3. Upload training data to HF Hub
cd scripts
python upload_datasets.py

# 4. Run training on HF Jobs
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
